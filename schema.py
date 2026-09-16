import re
from typing import Any
from urllib.parse import urlparse


POLICY_STATUSES = [
    "explicitly_disclosed",
    "broadly_disclosed",
    "implicitly_disclosed",
    "not_clearly_disclosed",
    "contradicted",
    "unknown",
]
TELEMETRY_STATUSES = [
    "observed",
    "not_observed",
    "insufficient_sample",
    "unsupported",
]
COMPARISONS = [
    "matched",
    "partially_matched",
    "policy_only",
    "observed_only",
    "possible_contradiction",
    "indeterminate",
]
SEVERITIES = ["informational", "low", "medium", "high", "critical"]
COUNT_FIELDS = [
    "matched",
    "partially_matched",
    "policy_only",
    "observed_only",
    "possible_contradictions",
    "indeterminate",
]
COMPARISON_TO_COUNT = {
    "matched": "matched",
    "partially_matched": "partially_matched",
    "policy_only": "policy_only",
    "observed_only": "observed_only",
    "possible_contradiction": "possible_contradictions",
    "indeterminate": "indeterminate",
}


REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "domain",
        "privacy_policy",
        "visit",
        "analysis",
        "findings",
        "important_limitations",
    ],
    "properties": {
        "domain": {"type": "string", "minLength": 1},
        "privacy_policy": {
            "type": "object",
            "additionalProperties": False,
            "required": ["url", "found", "applicable"],
            "properties": {
                "url": {"type": "string"},
                "found": {"type": "boolean"},
                "applicable": {"type": "boolean"},
            },
        },
        "visit": {
            "type": "object",
            "additionalProperties": False,
            "required": ["observed_at", "duration_seconds"],
            "properties": {
                "observed_at": {"type": ["string", "null"]},
                "duration_seconds": {"type": "integer", "minimum": 0},
            },
        },
        "analysis": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "overall_confidence", "counts"],
            "properties": {
                "summary": {"type": "string"},
                "overall_confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "counts": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": COUNT_FIELDS,
                    "properties": {
                        key: {"type": "integer", "minimum": 0}
                        for key in COUNT_FIELDS
                    },
                },
            },
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "behavior",
                    "category",
                    "description",
                    "policy",
                    "telemetry",
                    "comparison",
                    "severity",
                    "confidence",
                    "explanation",
                ],
                "properties": {
                    "behavior": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "policy": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["status", "evidence", "section"],
                        "properties": {
                            "status": {"type": "string", "enum": POLICY_STATUSES},
                            "evidence": {"type": "string"},
                            "section": {"type": "string"},
                        },
                    },
                    "telemetry": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["status", "evidence", "observation_count"],
                        "properties": {
                            "status": {"type": "string", "enum": TELEMETRY_STATUSES},
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                            "observation_count": {
                                "type": ["integer", "null"],
                                "minimum": 0,
                            },
                        },
                    },
                    "comparison": {"type": "string", "enum": COMPARISONS},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "explanation": {"type": "string"},
                },
            },
        },
        "important_limitations": {"type": "array", "items": {"type": "string"}},
    },
}


MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(\s*(?:https?://|www\.)", re.I)
HTML_LINK_RE = re.compile(r"<\s*a\b", re.I)
HTML_TAG_RE = re.compile(r"<\s*/?\s*[a-z][^>]*>", re.I)
MARKDOWN_FORMAT_RE = re.compile(
    r"(?:\*\*|`|(?m:^\s{0,3}#{1,6}\s)|(?m:^\s*(?:[-+*]|\d+\.)\s+))"
)
URL_RE = re.compile(r"https?://", re.I)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _object(value: Any, path: str, exact_keys: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    missing = exact_keys - set(value)
    extra = set(value) - exact_keys
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{path} has unexpected keys: {sorted(extra)}")
    return value


def _string(value: Any, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    if nonempty and not value:
        raise ValueError(f"{path} must not be empty")
    if "\\_" in value:
        raise ValueError(f"{path} contains a Markdown-escaped underscore")
    if MARKDOWN_LINK_RE.search(value):
        raise ValueError(f"{path} contains a Markdown link")
    if HTML_LINK_RE.search(value):
        raise ValueError(f"{path} contains an HTML link")
    if HTML_TAG_RE.search(value):
        raise ValueError(f"{path} contains HTML formatting")
    if MARKDOWN_FORMAT_RE.search(value):
        raise ValueError(f"{path} contains Markdown formatting")
    return value


def _raw_http_url(value: str, path: str, *, allow_empty: bool = False) -> None:
    _string(value, path)
    if not value and allow_empty:
        return
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{path} must contain one raw absolute HTTP(S) URL")
    if value.strip() != value or any(char in value for char in "[]()<>\n\r\t"):
        raise ValueError(f"{path} contains formatting instead of a raw URL")


def recompute_counts(findings: list[dict]) -> dict:
    counts = {key: 0 for key in COUNT_FIELDS}
    for finding in findings:
        comparison = finding.get("comparison")
        field = COMPARISON_TO_COUNT.get(comparison)
        if field:
            counts[field] += 1
    return counts


def _validate_finding(finding: Any, index: int, policy_applicable: bool) -> None:
    path = f"findings/{index}"
    finding = _object(
        finding,
        path,
        {
            "behavior",
            "category",
            "description",
            "policy",
            "telemetry",
            "comparison",
            "severity",
            "confidence",
            "explanation",
        },
    )
    _string(finding["behavior"], f"{path}/behavior", nonempty=True)
    _string(finding["category"], f"{path}/category", nonempty=True)
    description = _string(finding["description"], f"{path}/description")
    explanation = _string(finding["explanation"], f"{path}/explanation")
    if URL_RE.search(description) or URL_RE.search(explanation):
        raise ValueError(f"{path} description/explanation must not contain URLs")

    policy = _object(
        finding["policy"],
        f"{path}/policy",
        {"status", "evidence", "section"},
    )
    if policy["status"] not in POLICY_STATUSES:
        raise ValueError(f"{path}/policy/status has invalid value {policy['status']!r}")
    for field in ("evidence", "section"):
        value = _string(policy[field], f"{path}/policy/{field}")
        if URL_RE.search(value):
            raise ValueError(f"{path}/policy/{field} must not contain URLs")

    telemetry = _object(
        finding["telemetry"],
        f"{path}/telemetry",
        {"status", "evidence", "observation_count"},
    )
    if telemetry["status"] not in TELEMETRY_STATUSES:
        raise ValueError(
            f"{path}/telemetry/status has invalid value {telemetry['status']!r}"
        )
    if not isinstance(telemetry["evidence"], list):
        raise ValueError(f"{path}/telemetry/evidence must be an array")
    for evidence_index, evidence in enumerate(telemetry["evidence"]):
        value = _string(
            evidence,
            f"{path}/telemetry/evidence/{evidence_index}",
            nonempty=True,
        )
        if URL_RE.search(value):
            raise ValueError(
                f"{path}/telemetry/evidence/{evidence_index} must not contain URLs"
            )
    observation_count = telemetry["observation_count"]
    if observation_count is not None and (
        not _is_int(observation_count) or observation_count < 0
    ):
        raise ValueError(
            f"{path}/telemetry/observation_count must be null or a non-negative integer"
        )
    if telemetry["status"] == "observed" and (
        not _is_int(observation_count) or observation_count < 1
    ):
        raise ValueError(
            f"{path}/telemetry/observation_count must be positive for observed behavior"
        )
    if telemetry["status"] == "observed" and not telemetry["evidence"]:
        raise ValueError(f"{path}/telemetry/evidence must not be empty for observed behavior")

    if policy["status"] in {
        "explicitly_disclosed",
        "broadly_disclosed",
        "implicitly_disclosed",
        "contradicted",
    } and (not policy["evidence"].strip() or not policy["section"].strip()):
        raise ValueError(
            f"{path}/policy requires evidence and section for disclosure status {policy['status']}"
        )

    comparison = finding["comparison"]
    if comparison not in COMPARISONS:
        raise ValueError(f"{path}/comparison has invalid value {comparison!r}")
    if finding["severity"] not in SEVERITIES:
        raise ValueError(f"{path}/severity has invalid value {finding['severity']!r}")
    if not _is_number(finding["confidence"]) or not 0 <= finding["confidence"] <= 1:
        raise ValueError(f"{path}/confidence must be between 0 and 1")

    if not policy_applicable:
        if comparison != "indeterminate" or policy["status"] != "unknown":
            raise ValueError(
                f"{path} must be indeterminate with unknown policy status when no applicable policy exists"
            )
    elif comparison == "matched":
        if telemetry["status"] != "observed" or policy["status"] != "explicitly_disclosed":
            raise ValueError(
                f"{path} matched requires observed telemetry and explicit disclosure"
            )
    elif comparison == "partially_matched":
        if telemetry["status"] != "observed" or policy["status"] not in {
            "broadly_disclosed",
            "implicitly_disclosed",
        }:
            raise ValueError(
                f"{path} partially_matched requires observed telemetry and broad or implicit disclosure"
            )
    elif comparison == "observed_only":
        if telemetry["status"] != "observed" or policy["status"] != "not_clearly_disclosed":
            raise ValueError(
                f"{path} observed_only requires observed telemetry and no clear disclosure"
            )
    elif comparison == "possible_contradiction":
        if telemetry["status"] != "observed" or policy["status"] != "contradicted":
            raise ValueError(
                f"{path} possible_contradiction requires observed telemetry and a contradicted policy statement"
            )
    elif comparison == "policy_only":
        if telemetry["status"] not in {
            "not_observed",
            "insufficient_sample",
            "unsupported",
        } or policy["status"] not in {
            "explicitly_disclosed",
            "broadly_disclosed",
            "implicitly_disclosed",
        }:
            raise ValueError(
                f"{path} policy_only requires a disclosure and non-observed or unsupported telemetry"
            )


def validate_report(report: dict) -> dict:
    report = _object(
        report,
        "<root>",
        {
            "domain",
            "privacy_policy",
            "visit",
            "analysis",
            "findings",
            "important_limitations",
        },
    )
    _string(report["domain"], "domain", nonempty=True)

    privacy_policy = _object(
        report["privacy_policy"],
        "privacy_policy",
        {"url", "found", "applicable"},
    )
    if not isinstance(privacy_policy["found"], bool) or not isinstance(
        privacy_policy["applicable"], bool
    ):
        raise ValueError("privacy_policy.found/applicable must be booleans")
    if privacy_policy["applicable"] and not privacy_policy["found"]:
        raise ValueError("privacy_policy.applicable cannot be true when found is false")
    _raw_http_url(
        privacy_policy["url"],
        "privacy_policy.url",
        allow_empty=not privacy_policy["found"],
    )
    if not privacy_policy["found"] and privacy_policy["url"]:
        raise ValueError("privacy_policy.url must be empty when found is false")

    visit = _object(
        report["visit"], "visit", {"observed_at", "duration_seconds"}
    )
    if visit["observed_at"] is not None:
        _string(visit["observed_at"], "visit.observed_at", nonempty=True)
    if not _is_int(visit["duration_seconds"]) or visit["duration_seconds"] < 0:
        raise ValueError("visit.duration_seconds must be a non-negative integer")

    analysis = _object(
        report["analysis"],
        "analysis",
        {"summary", "overall_confidence", "counts"},
    )
    summary = _string(analysis["summary"], "analysis.summary")
    if URL_RE.search(summary):
        raise ValueError("analysis.summary must not contain URLs")
    if not _is_number(analysis["overall_confidence"]) or not (
        0 <= analysis["overall_confidence"] <= 1
    ):
        raise ValueError("analysis.overall_confidence must be between 0 and 1")
    counts = _object(analysis["counts"], "analysis.counts", set(COUNT_FIELDS))
    for key, value in counts.items():
        if not _is_int(value) or value < 0:
            raise ValueError(f"analysis.counts.{key} must be a non-negative integer")

    if not isinstance(report["findings"], list):
        raise ValueError("findings must be an array")
    policy_applicable = privacy_policy["found"] and privacy_policy["applicable"]
    for index, finding in enumerate(report["findings"]):
        _validate_finding(finding, index, policy_applicable)

    calculated = recompute_counts(report["findings"])
    if counts != calculated:
        raise ValueError(
            f"analysis.counts mismatch: expected {calculated}, got {counts}"
        )

    if not isinstance(report["important_limitations"], list):
        raise ValueError("important_limitations must be an array")
    for index, limitation in enumerate(report["important_limitations"]):
        value = _string(
            limitation,
            f"important_limitations/{index}",
            nonempty=True,
        )
        if URL_RE.search(value):
            raise ValueError(
                f"important_limitations/{index} must not contain URLs"
            )
    return report
