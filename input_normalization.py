import copy
import datetime as dt
from typing import Any, Optional
from urllib.parse import urlparse

from telemetry import (
    PAGE_COUNT_FIELDS,
    SECURITY_FIELDS,
    SNAPSHOT_SCHEMA,
    extract_snapshot,
    validate_snapshot_shape,
)


SEEN_BEHAVIOR_KEYS = (
    "site",
    "observation",
    "thirdPartyHosts",
    "trackers",
    "signals",
    "page",
    "security",
    "interest",
)


def _iso_from_milliseconds(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        return dt.datetime.fromtimestamp(
            value / 1000.0, tz=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _normalize_http_url(value: Any, *, allow_empty: bool = False) -> Optional[str]:
    if value is None or value == "":
        return "" if allow_empty else None
    if not isinstance(value, str):
        raise ValueError("URL values must be strings or null")
    value = value.strip()
    if not value:
        return "" if allow_empty else None
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid HTTP(S) URL: {value!r}")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not accepted")
    return value


def _minimal_redacted_document() -> dict:
    return {
        "format": "veilance.redacted-html.v1",
        "html": "",
        "truncated": False,
        "originalElementCount": 0,
        "serializedChars": 0,
        "redaction": {
            "textNodesRedacted": 0,
            "attributesRemoved": 0,
            "urlsReduced": 0,
            "privateUrlsRemoved": 0,
            "inlineScriptsRedacted": 0,
            "styleBlocksRedacted": 0,
            "formControlsRedacted": 0,
            "commentsRemoved": 0,
            "opaqueNodesRedacted": 0,
            "nodesOmitted": 0,
        },
        "evidence": {
            "resourceHosts": [],
            "inlineScriptHints": {
                "canvas": 0,
                "webgl": 0,
                "webgpu": 0,
                "audio": 0,
                "fonts": 0,
                "navigator": 0,
                "screen": 0,
                "webrtc": 0,
                "advertising": 0,
                "antiBlocking": 0,
            },
            "domMarkers": {
                "advertising": 0,
                "consent": 0,
                "antiBlocking": 0,
                "tracking": 0,
            },
        },
    }


def snapshot_from_seen_behavior(record: dict) -> dict:
    """Reconstruct the exact v2 envelope from the public analysis input shape."""

    seen = record.get("seen_behavior")
    visit = record.get("visit") or {}
    if not isinstance(seen, dict) or not isinstance(visit, dict):
        raise ValueError("seen_behavior and visit must be objects")

    required = set(SEEN_BEHAVIOR_KEYS)
    missing = required - set(seen)
    if missing:
        raise ValueError(f"seen_behavior missing keys: {sorted(missing)}")

    snapshot = {
        "schemaVersion": SNAPSHOT_SCHEMA,
        "eventId": str(visit.get("snapshot_id") or "unknown-snapshot"),
        "extensionVersion": str(visit.get("extension_version") or "unknown"),
        **{key: copy.deepcopy(seen[key]) for key in SEEN_BEHAVIOR_KEYS},
        "redactedDocument": copy.deepcopy(
            record.get("redactedDocument") or _minimal_redacted_document()
        ),
    }

    duration = visit.get("duration_seconds")
    if duration is not None:
        snapshot["observation"]["durationSeconds"] = duration
    return validate_snapshot_shape(snapshot)


def snapshot_from_connector_payload(record: dict) -> dict:
    """Accept the exact object currently passed to privacy_policy_comparison().

    That contract keeps request counts under ``seen_behavior.observations`` and
    places hosts, trackers, signals, page data, and detections at the root.
    Missing security/interest fields are filled only to form an internal v2
    envelope; those inferred placeholders are not included in the model input.
    """

    seen = record.get("seen_behavior")
    visit = record.get("visit")
    if not isinstance(seen, dict) or not isinstance(visit, dict):
        raise ValueError("seen_behavior and visit must be objects")
    observation = seen.get("observations")
    if not isinstance(observation, dict):
        observation = seen.get("observation")
    if not isinstance(observation, dict):
        raise ValueError(
            "connector input must contain seen_behavior.observations"
        )

    domain_url = _normalize_http_url(record.get("domain_url"))
    parsed = urlparse(domain_url)
    observation = copy.deepcopy(observation)
    observation["durationSeconds"] = visit.get(
        "duration_seconds", observation.get("durationSeconds", 0)
    )
    for field in ("totalRequests", "firstPartyRequests", "thirdPartyRequests"):
        observation.setdefault(field, 0)

    page_input = record.get("page") if isinstance(record.get("page"), dict) else {}
    page = {
        field: page_input.get(field, 0)
        for field in PAGE_COUNT_FIELDS
    }
    page["serviceWorkerControlled"] = page_input.get(
        "serviceWorkerControlled", False
    )

    security_input = (
        record.get("security") if isinstance(record.get("security"), dict) else {}
    )
    security = {field: security_input.get(field, False) for field in SECURITY_FIELDS}

    detections = record.get("detections")
    if not isinstance(detections, list):
        detections = []
    reasons = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        detection_type = detection.get("type")
        severity = detection.get("severity")
        if isinstance(detection_type, str) and isinstance(severity, str):
            reasons.append(
                {"id": detection_type, "severity": severity, "points": 0}
            )

    snapshot = {
        "schemaVersion": SNAPSHOT_SCHEMA,
        "eventId": str(visit.get("snapshot_id") or "unknown-snapshot"),
        "extensionVersion": str(visit.get("extension_version") or "unknown"),
        "site": {
            "hostname": parsed.hostname,
            "https": parsed.scheme == "https",
        },
        "observation": observation,
        "thirdPartyHosts": copy.deepcopy(record.get("thirdPartyHosts") or []),
        "trackers": copy.deepcopy(record.get("trackers") or []),
        "signals": copy.deepcopy(record.get("signals") or []),
        "page": page,
        "security": security,
        "interest": {
            "score": 0,
            "level": "not_supplied",
            "minimumScore": 0,
            "eligible": False,
            "reasons": reasons,
        },
        "redactedDocument": _minimal_redacted_document(),
    }
    return validate_snapshot_shape(snapshot)


def normalize_runtime_input(
    record: dict,
    *,
    domain_url_override: Optional[str] = None,
    policy_url_override: Optional[str] = None,
) -> dict:
    """Normalize every supported API payload into one retrieval/analysis record."""

    if not isinstance(record, dict):
        raise ValueError("input must be a JSON object")

    try:
        snapshot = extract_snapshot(record)
    except ValueError:
        if isinstance(record.get("seen_behavior"), dict):
            if "observations" in record["seen_behavior"]:
                snapshot = snapshot_from_connector_payload(record)
            else:
                snapshot = snapshot_from_seen_behavior(record)
        else:
            raise

    site = snapshot["site"]
    default_domain = (
        ("https" if site["https"] else "http") + "://" + site["hostname"]
    )
    domain_url = _normalize_http_url(
        domain_url_override or record.get("domain_url") or default_domain
    )

    supplied_document = record.get("policy_document")
    document_url = (
        supplied_document.get("url")
        if isinstance(supplied_document, dict)
        else None
    )
    supplied_policy_url = (
        policy_url_override
        if policy_url_override is not None
        else record.get("privacy_policy_url", document_url)
    )
    try:
        privacy_policy_url = _normalize_http_url(
            supplied_policy_url,
            allow_empty=True,
        )
    except ValueError:
        # A malformed/outdated supplied URL must not prevent first-party policy
        # discovery. Preserve the string so PolicyRetriever can record the failed
        # supplied candidate and continue with footer/common-path/search discovery.
        if not isinstance(supplied_policy_url, str):
            raise
        privacy_policy_url = supplied_policy_url.strip()
    privacy_policy_url = privacy_policy_url or None

    visit_input = record.get("visit") if isinstance(record.get("visit"), dict) else {}
    observed_at = visit_input.get("observed_at")
    if isinstance(observed_at, (int, float)) and not isinstance(observed_at, bool):
        observed_at = _iso_from_milliseconds(observed_at)
    if not isinstance(observed_at, str) or not observed_at:
        observed_at = snapshot["observation"].get("observedAt")
    if not isinstance(observed_at, str) or not observed_at:
        observed_at = _iso_from_milliseconds(record.get("createdAt"))

    visit = {
        "snapshot_id": str(
            visit_input.get("snapshot_id")
            or record.get("snapshotId")
            or snapshot["eventId"]
        ),
        "observed_at": observed_at,
        "duration_seconds": snapshot["observation"]["durationSeconds"],
        "extension_version": snapshot["extensionVersion"],
    }

    return {
        "domain_url": domain_url,
        "privacy_policy_url": privacy_policy_url,
        "visit": visit,
        "snapshot": snapshot,
        "supplied_policy_document": supplied_document,
    }


def validate_policy_document(document: dict) -> dict:
    if not isinstance(document, dict):
        raise ValueError("policy_document must be an object")
    for field in ("url", "found", "applicable", "sections"):
        if field not in document:
            raise ValueError(f"policy_document missing {field}")
    if not isinstance(document["url"], str):
        raise ValueError("policy_document.url must be a string")
    if not isinstance(document["found"], bool):
        raise ValueError("policy_document.found must be a boolean")
    if not isinstance(document["applicable"], bool):
        raise ValueError("policy_document.applicable must be a boolean")
    if document["applicable"] and not document["found"]:
        raise ValueError("an applicable policy_document must also be found")
    if document["found"]:
        _normalize_http_url(document["url"])
    elif document["url"]:
        raise ValueError("policy_document.url must be empty when no policy was found")
    if not isinstance(document["sections"], list):
        raise ValueError("policy_document.sections must be an array")
    for index, section in enumerate(document["sections"]):
        if not isinstance(section, dict):
            raise ValueError(f"policy_document.sections[{index}] must be an object")
        if set(section) != {"heading", "text"}:
            raise ValueError(
                f"policy_document.sections[{index}] must contain heading and text only"
            )
        if not isinstance(section["heading"], str) or not isinstance(section["text"], str):
            raise ValueError(
                f"policy_document.sections[{index}] heading/text must be strings"
            )
        if "EXAMPLE PLACEHOLDER" in section["text"].upper():
            raise ValueError(
                f"policy_document.sections[{index}] contains placeholder policy text"
            )
    if document["found"] and document["applicable"] and not document["sections"]:
        raise ValueError("an applicable policy_document must contain extracted sections")
    limitations = document.get("limitations", [])
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise ValueError("policy_document.limitations must be an array of strings")
    return document


def _connector_detections(snapshot: dict) -> list[dict]:
    detections = []
    for reason in snapshot["interest"].get("reasons", []):
        if not isinstance(reason, dict):
            continue
        reason_id = reason.get("id")
        severity = reason.get("severity")
        if isinstance(reason_id, str) and isinstance(severity, str):
            detections.append(
                {
                    "type": reason_id.replace("detection.veilance-json-", ""),
                    "severity": severity,
                }
            )
    return detections


def build_analysis_input(runtime: dict, policy_document: dict) -> dict:
    policy_document = validate_policy_document(copy.deepcopy(policy_document))
    snapshot = validate_snapshot_shape(runtime["snapshot"])
    compact_policy = {
        key: copy.deepcopy(policy_document[key])
        for key in ("url", "found", "applicable", "sections")
    }
    if policy_document.get("limitations"):
        compact_policy["limitations"] = copy.deepcopy(policy_document["limitations"])

    observation = snapshot["observation"]
    return {
        "domain_url": runtime["domain_url"],
        "privacy_policy_url": (
            policy_document["url"]
            if policy_document["found"] and policy_document["applicable"]
            else None
        ),
        "visit": copy.deepcopy(runtime["visit"]),
        "seen_behavior": {
            "observations": {
                "totalRequests": observation["totalRequests"],
                "firstPartyRequests": observation["firstPartyRequests"],
                "thirdPartyRequests": observation["thirdPartyRequests"],
            }
        },
        "thirdPartyHosts": copy.deepcopy(snapshot["thirdPartyHosts"]),
        "trackers": copy.deepcopy(snapshot["trackers"]),
        "signals": copy.deepcopy(snapshot["signals"]),
        "page": copy.deepcopy(snapshot["page"]),
        "detections": _connector_detections(snapshot),
        "policy_document": compact_policy,
    }
