from jsonschema import Draft202012Validator


POLICY_STATUSES = [
    "explicitly_disclosed", "broadly_disclosed", "partially_disclosed",
    "not_clearly_disclosed", "contradicted", "not_applicable", "indeterminate"
]
TELEMETRY_STATUSES = ["observed", "not_observed", "indeterminate"]
COMPARISONS = [
    "matched", "partially_matched", "policy_only", "observed_only",
    "possible_contradiction", "indeterminate"
]
SEVERITIES = ["informational", "low", "medium", "high"]
REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["domain", "privacy_policy", "visit", "analysis", "findings", "important_limitations"],
    "properties": {
        "domain": {"type": "string", "minLength": 1},
        "privacy_policy": {
            "type": "object", "additionalProperties": False,
            "required": ["url", "found", "applicable"],
            "properties": {
                "url": {"type": ["string", "null"]},
                "found": {"type": "boolean"},
                "applicable": {"type": "boolean"}
            }
        },
        "visit": {
            "type": "object", "additionalProperties": False,
            "required": ["observed_at", "duration_seconds"],
            "properties": {
                "observed_at": {"type": ["string", "null"]},
                "duration_seconds": {"type": "integer", "minimum": 0}
            }
        },
        "analysis": {
            "type": "object", "additionalProperties": False,
            "required": ["summary", "overall_confidence", "counts"],
            "properties": {
                "summary": {"type": "string"},
                "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "counts": {
                    "type": "object", "additionalProperties": False,
                    "required": ["matched", "partially_matched", "policy_only", "observed_only", "possible_contradictions", "indeterminate"],
                    "properties": {k: {"type": "integer", "minimum": 0} for k in [
                        "matched", "partially_matched", "policy_only", "observed_only", "possible_contradictions", "indeterminate"
                    ]}
                }
            }
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["behavior", "category", "description", "policy", "telemetry", "comparison", "severity", "confidence", "explanation"],
                "properties": {
                    "behavior": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "policy": {
                        "type": "object", "additionalProperties": False,
                        "required": ["status", "evidence", "section"],
                        "properties": {
                            "status": {"enum": POLICY_STATUSES},
                            "evidence": {"type": "string"},
                            "section": {"type": "string"}
                        }
                    },
                    "telemetry": {
                        "type": "object", "additionalProperties": False,
                        "required": ["status", "evidence", "observation_count"],
                        "properties": {
                            "status": {"enum": TELEMETRY_STATUSES},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                            "observation_count": {"type": "integer", "minimum": 0}
                        }
                    },
                    "comparison": {"enum": COMPARISONS},
                    "severity": {"enum": SEVERITIES},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "explanation": {"type": "string"}
                }
            }
        },
        "important_limitations": {
            "type": "array", "items": {"type": "string"}
        }
    }
}
VALIDATOR = Draft202012Validator(REPORT_SCHEMA)


def validate_report(report):
    errors = sorted(VALIDATOR.iter_errors(report), key=lambda e: list(e.path))
    if errors:
        raise ValueError("\n".join(
            f"{'/'.join(map(str,e.path)) or '<root>'}: {e.message}" for e in errors
        ))

    counts = {k: 0 for k in ["matched", "partially_matched", "policy_only", "observed_only", "possible_contradictions", "indeterminate"]}
    mapping = {"possible_contradiction": "possible_contradictions"}
    for finding in report["findings"]:
        key = mapping.get(finding["comparison"], finding["comparison"])
        counts[key] += 1
    if report["analysis"]["counts"] != counts:
        raise ValueError(f"analysis.counts mismatch: expected {counts}")
