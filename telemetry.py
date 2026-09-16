import gzip
import json
from pathlib import Path
from typing import Any


SNAPSHOT_SCHEMA = "veilance.telemetry-snapshot.v2"
BATCH_SCHEMA = "veilance.telemetry-snapshot-batch.v1"
EXPECTED_SNAPSHOT_KEYS = {
    "schemaVersion",
    "eventId",
    "extensionVersion",
    "site",
    "observation",
    "thirdPartyHosts",
    "trackers",
    "signals",
    "page",
    "security",
    "interest",
    "redactedDocument",
}

PAGE_COUNT_FIELDS = {
    "scriptCount",
    "thirdPartyScriptCount",
    "iframeCount",
    "thirdPartyIframeCount",
    "accessibleCookieCount",
    "localStorageKeyCount",
    "sessionStorageKeyCount",
    "indexedDbCount",
    "cacheCount",
}
SECURITY_FIELDS = {
    "contentSecurityPolicy",
    "strictTransportSecurity",
    "permissionsPolicy",
    "referrerPolicy",
    "xFrameOptions",
    "crossOriginOpenerPolicy",
    "crossOriginResourcePolicy",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _require_nonnegative_int(value: Any, path: str) -> None:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")


def validate_snapshot_shape(snapshot: dict, *, strict_top_level: bool = True) -> dict:
    """Validate the fields used by the real veilance.telemetry-snapshot.v2 payload.

    Nested objects may gain additive fields in future extension releases, but known
    fields retain their real names and types. This deliberately rejects the old
    synthetic-only aliases ``types`` and ``serviceWorker``.
    """

    snapshot = _require_object(snapshot, "telemetry snapshot")
    if snapshot.get("schemaVersion") != SNAPSHOT_SCHEMA:
        raise ValueError(
            f"expected {SNAPSHOT_SCHEMA}, got {snapshot.get('schemaVersion')!r}"
        )

    missing = EXPECTED_SNAPSHOT_KEYS - set(snapshot)
    extra = set(snapshot) - EXPECTED_SNAPSHOT_KEYS
    if missing:
        raise ValueError(f"snapshot missing keys: {sorted(missing)}")
    if strict_top_level and extra:
        raise ValueError(f"snapshot has unexpected keys: {sorted(extra)}")

    if not isinstance(snapshot.get("eventId"), str) or not snapshot["eventId"]:
        raise ValueError("eventId must be a non-empty string")
    if not isinstance(snapshot.get("extensionVersion"), str):
        raise ValueError("extensionVersion must be a string")

    site = _require_object(snapshot.get("site"), "site")
    if not isinstance(site.get("hostname"), str) or not site["hostname"]:
        raise ValueError("site.hostname must be a non-empty string")
    if not isinstance(site.get("https"), bool):
        raise ValueError("site.https must be a boolean")

    observation = _require_object(snapshot.get("observation"), "observation")
    for field in (
        "durationSeconds",
        "totalRequests",
        "firstPartyRequests",
        "thirdPartyRequests",
    ):
        _require_nonnegative_int(observation.get(field), f"observation.{field}")

    for index, host in enumerate(
        _require_list(snapshot.get("thirdPartyHosts"), "thirdPartyHosts")
    ):
        host = _require_object(host, f"thirdPartyHosts[{index}]")
        if not isinstance(host.get("host"), str) or not host["host"]:
            raise ValueError(f"thirdPartyHosts[{index}].host must be a string")
        _require_nonnegative_int(
            host.get("requests"), f"thirdPartyHosts[{index}].requests"
        )
        if "types" in host:
            raise ValueError(
                f"thirdPartyHosts[{index}] uses synthetic key 'types'; "
                "veilance.telemetry-snapshot.v2 uses 'resourceTypes'"
            )
        resource_types = _require_object(
            host.get("resourceTypes"),
            f"thirdPartyHosts[{index}].resourceTypes",
        )
        for resource_type, count in resource_types.items():
            if not isinstance(resource_type, str) or not resource_type:
                raise ValueError(
                    f"thirdPartyHosts[{index}].resourceTypes keys must be strings"
                )
            _require_nonnegative_int(
                count,
                f"thirdPartyHosts[{index}].resourceTypes.{resource_type}",
            )

    for index, tracker in enumerate(
        _require_list(snapshot.get("trackers"), "trackers")
    ):
        tracker = _require_object(tracker, f"trackers[{index}]")
        for field in ("id", "category"):
            if not isinstance(tracker.get(field), str) or not tracker[field]:
                raise ValueError(f"trackers[{index}].{field} must be a string")
        _require_nonnegative_int(
            tracker.get("requests"), f"trackers[{index}].requests"
        )

    for index, signal in enumerate(
        _require_list(snapshot.get("signals"), "signals")
    ):
        signal = _require_object(signal, f"signals[{index}]")
        for field in ("indicatorId", "api", "action"):
            if not isinstance(signal.get(field), str) or not signal[field]:
                raise ValueError(f"signals[{index}].{field} must be a string")
        _require_nonnegative_int(signal.get("count"), f"signals[{index}].count")

    page = _require_object(snapshot.get("page"), "page")
    if "serviceWorker" in page:
        raise ValueError(
            "page uses synthetic key 'serviceWorker'; "
            "veilance.telemetry-snapshot.v2 uses 'serviceWorkerControlled'"
        )
    for field in PAGE_COUNT_FIELDS:
        _require_nonnegative_int(page.get(field), f"page.{field}")
    if not isinstance(page.get("serviceWorkerControlled"), bool):
        raise ValueError("page.serviceWorkerControlled must be a boolean")

    security = _require_object(snapshot.get("security"), "security")
    for field in SECURITY_FIELDS:
        if not isinstance(security.get(field), bool):
            raise ValueError(f"security.{field} must be a boolean")

    interest = _require_object(snapshot.get("interest"), "interest")
    _require_nonnegative_int(interest.get("score"), "interest.score")
    _require_nonnegative_int(interest.get("minimumScore"), "interest.minimumScore")
    if not isinstance(interest.get("level"), str):
        raise ValueError("interest.level must be a string")
    if not isinstance(interest.get("eligible"), bool):
        raise ValueError("interest.eligible must be a boolean")
    for index, reason in enumerate(_require_list(interest.get("reasons"), "interest.reasons")):
        reason = _require_object(reason, f"interest.reasons[{index}]")
        if not isinstance(reason.get("id"), str) or not reason["id"]:
            raise ValueError(f"interest.reasons[{index}].id must be a string")
        if not isinstance(reason.get("severity"), str):
            raise ValueError(f"interest.reasons[{index}].severity must be a string")
        _require_nonnegative_int(
            reason.get("points"), f"interest.reasons[{index}].points"
        )

    _require_object(snapshot.get("redactedDocument"), "redactedDocument")
    return snapshot


def extract_snapshot(record: dict) -> dict:
    """Accept an exact snapshot, an upload envelope, or a training/runtime record."""

    if not isinstance(record, dict):
        raise ValueError("input must be a JSON object")
    if record.get("schemaVersion") == SNAPSHOT_SCHEMA:
        return validate_snapshot_shape(record)
    for key in ("telemetry", "payload"):
        candidate = record.get(key)
        if isinstance(candidate, dict) and candidate.get("schemaVersion") == SNAPSHOT_SCHEMA:
            return validate_snapshot_shape(candidate)
    raise ValueError(
        "input does not contain a veilance.telemetry-snapshot.v2 object "
        "at the root, telemetry, or payload"
    )


def load_telemetry_file(path):
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("telemetry file must contain a JSON object")
    if obj.get("schemaVersion") == BATCH_SCHEMA:
        observations = obj.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("batch has no observations")
        return [validate_snapshot_shape(item) for item in observations]
    return [extract_snapshot(obj)]
