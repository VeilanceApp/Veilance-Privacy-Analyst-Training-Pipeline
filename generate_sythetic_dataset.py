import argparse
import copy
import datetime
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


MODE_STANDALONE = "standalone_report"
MODE_DATABASE = "database_report"
MODE_TRAINING = "training_row"

SNAPSHOT_SCHEMA = "veilance.telemetry-snapshot.v2"

COUNT_FIELDS = {
    "matched": "matched",
    "partially_matched": "partially_matched",
    "policy_only": "policy_only",
    "observed_only": "observed_only",
    "possible_contradiction": "possible_contradictions",
    "indeterminate": "indeterminate",
}

ALLOWED_COMPARISONS = set(COUNT_FIELDS)
ALLOWED_POLICY_STATUSES = {
    "explicitly_disclosed",
    "broadly_disclosed",
    "partially_disclosed",
    "not_clearly_disclosed",
    "contradicted",
    "not_applicable",
    "indeterminate",
}
ALLOWED_TELEMETRY_STATUSES = {"observed", "not_observed", "indeterminate"}
ALLOWED_SEVERITIES = {"informational", "low", "medium", "high"}

# This recognizes Veilance API-like evidence such as:
#   Navigator.read-user-agent observed 5 times
# It does NOT parse or cast the number from the sentence.
API_EVIDENCE_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z0-9_-]+)\s+observed\b",
    re.IGNORECASE,
)

HOST_RE = re.compile(
    r"(?<![A-Za-z0-9_-])([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?![A-Za-z0-9_-])"
)


# ============================================================
# Basic helpers
# ============================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def json_dumps_stable(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_float(value: Any) -> bool:
    return isinstance(value, float)


def is_number(value: Any) -> bool:
    return is_integer(value) or is_float(value)


def same_type(original: Any, mutated: Any) -> bool:
    return type(original) is type(mutated)


def clamp(value: Any, minimum: Any, maximum: Any) -> Any:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def host_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname:
        return parsed.hostname
    return url.replace("https://", "").replace("http://", "").split("/")[0]


def synthetic_domain(source_domain: str, variant_index: int, seed: str) -> str:
    token = stable_hash(f"{source_domain}:{variant_index}:{seed}")
    return f"https://site-{token}.example"


def synthetic_policy_url(domain: str) -> str:
    return domain.rstrip("/") + "/privacy"


# ============================================================
# Input loading
# ============================================================

def iter_json_file(path: Path) -> Iterable[dict]:
    data = load_json(path)

    if isinstance(data, dict):
        yield data
        return

    if isinstance(data, list):
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: element {index} is not an object")
            yield item
        return

    raise ValueError(f"{path}: top-level JSON must be an object or array")


def iter_jsonl_file(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield item


def collect_input_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("Input file must be .json or .jsonl")
        return [path]

    if path.is_dir():
        return sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in {".json", ".jsonl"}
        )

    raise ValueError(f"Input path does not exist: {path}")


def load_records(path: Path) -> List[dict]:
    records = []
    for file in collect_input_files(path):
        if file.suffix.lower() == ".jsonl":
            records.extend(iter_jsonl_file(file))
        else:
            records.extend(iter_json_file(file))
    return records


# ============================================================
# Generic path helpers for safe configured mutations
# ============================================================

def split_path(path: str) -> List[str]:
    return path.split(".") if path else []


def find_matching_nodes(root: Any, path: str) -> List[Tuple[Any, Any, Any]]:
    parts = split_path(path)
    results = []

    def walk(current: Any, parent: Any, parent_key: Any, remaining: List[str]) -> None:
        if not remaining:
            results.append((parent, parent_key, current))
            return

        token = remaining[0]
        rest = remaining[1:]

        if token == "*":
            if isinstance(current, list):
                for index, item in enumerate(current):
                    walk(item, current, index, rest)
            elif isinstance(current, dict):
                for key, item in current.items():
                    walk(item, current, key, rest)
            return

        if isinstance(current, dict) and token in current:
            walk(current[token], current, token, rest)

    walk(root, None, None, parts)
    return results


def mutate_integer(value: int, rule: dict, rng: random.Random) -> int:
    strategy = rule.get("strategy")

    if strategy == "jitter":
        spread = rule.get("spread", 0.25)
        if not is_number(spread):
            raise ValueError("integer jitter spread must be numeric")

        delta = max(1, round(abs(value) * spread))
        low = value - delta
        high = value + delta

        minimum = rule.get("min")
        maximum = rule.get("max")

        if is_integer(minimum):
            low = max(low, minimum)
        if is_integer(maximum):
            high = min(high, maximum)
        if high < low:
            return value
        return rng.randint(low, high)

    if strategy == "choice":
        values = [candidate for candidate in rule.get("values", []) if type(candidate) is type(value)]
        if values:
            return rng.choice(values)

    return value


def mutate_float(value: float, rule: dict, rng: random.Random) -> float:
    strategy = rule.get("strategy")

    if strategy == "jitter":
        spread = rule.get("spread", 0.05)
        if not is_number(spread):
            raise ValueError("float jitter spread must be numeric")

        result = value + rng.uniform(-spread, spread)
        minimum = rule.get("min")
        maximum = rule.get("max")

        if is_number(minimum):
            result = max(result, minimum)
        if is_number(maximum):
            result = min(result, maximum)

        precision = rule.get("precision")
        if is_integer(precision):
            result = round(result, precision)
        return result

    if strategy == "choice":
        values = [candidate for candidate in rule.get("values", []) if type(candidate) is type(value)]
        if values:
            return rng.choice(values)

    return value


def mutate_string(value: str, rule: dict, rng: random.Random) -> str:
    strategy = rule.get("strategy")

    if strategy == "choice":
        values = [candidate for candidate in rule.get("values", []) if isinstance(candidate, str)]
        if values:
            return rng.choice(values)

    if strategy == "template":
        templates = [template for template in rule.get("templates", []) if isinstance(template, str)]
        if templates:
            return rng.choice(templates).replace("{original}", value)

    return value


def mutate_value(value: Any, rule: dict, rng: random.Random) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        if rule.get("strategy") == "choice":
            values = [candidate for candidate in rule.get("values", []) if type(candidate) is type(value)]
            if values:
                return rng.choice(values)
        return value

    if is_integer(value):
        return mutate_integer(value, rule, rng)
    if is_float(value):
        return mutate_float(value, rule, rng)
    if isinstance(value, str):
        return mutate_string(value, rule, rng)
    return value


def apply_mutation_rule(root: dict, path: str, rule: dict, rng: random.Random) -> int:
    probability = rule.get("probability", 1.0)
    if not is_number(probability):
        raise ValueError(f"{path}: probability must be numeric")

    changed = 0
    for parent, key, current in find_matching_nodes(root, path):
        if parent is None:
            continue
        if rng.random() > probability:
            continue

        mutated = mutate_value(current, rule, rng)
        if not same_type(current, mutated):
            raise TypeError(
                f"{path}: mutation changed {type(current).__name__} to {type(mutated).__name__}"
            )
        if mutated != current:
            parent[key] = mutated
            changed += 1
    return changed


def apply_report_mutations(report: dict, rules: dict, rng: random.Random) -> int:
    mutations = rules.get("report_mutations", {})
    if not isinstance(mutations, dict):
        raise ValueError("report_mutations must be an object")

    changed = 0
    for path, rule in mutations.items():
        if isinstance(rule, dict):
            changed += apply_mutation_rule(report, path, rule, rng)
    return changed


# ============================================================
# Source-shape detection
# ============================================================

def is_report(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("domain"), str)
        and isinstance(value.get("analysis"), dict)
        and isinstance(value.get("findings"), list)
    )


def detect_record_mode(record: dict) -> str:
    if (
        isinstance(record.get("telemetry"), dict)
        and isinstance(record.get("policy_document"), dict)
        and is_report(record.get("expected"))
    ):
        return MODE_TRAINING

    if is_report(record.get("policy_raw_results")):
        return MODE_DATABASE

    if is_report(record):
        return MODE_STANDALONE

    raise ValueError(
        "Unsupported record shape. Expected a standalone Veilance report, "
        "a database record containing policy_raw_results, or a prepared "
        "{telemetry, policy_document, expected} training row."
    )


def get_source_report(record: dict, mode: str) -> dict:
    if mode == MODE_STANDALONE:
        return record
    if mode == MODE_DATABASE:
        return record["policy_raw_results"]
    if mode == MODE_TRAINING:
        return record["expected"]
    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Report normalization
# ============================================================

def normalize_policy_status(status: Any) -> str:
    if status in ALLOWED_POLICY_STATUSES:
        return status
    if status == "unknown":
        return "indeterminate"
    if status == "implicitly_disclosed":
        return "broadly_disclosed"
    return "indeterminate"


def normalize_comparison(value: Any, policy_applicable: bool) -> str:
    if value in ALLOWED_COMPARISONS:
        if not policy_applicable:
            return "indeterminate"
        return value
    return "indeterminate" if not policy_applicable else "partially_matched"


def normalize_severity(value: Any) -> str:
    if value in ALLOWED_SEVERITIES:
        return value
    return "low"


def normalize_confidence(value: Any) -> float:
    if is_float(value):
        return clamp(value, 0.0, 1.0)
    if is_integer(value):
        if value <= 0:
            return 0.0
        return 1.0
    return 0.75


def normalize_report(source_report: dict) -> dict:
    report = deep_copy(source_report)

    domain = report.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError("report.domain must be a non-empty string")

    privacy_policy = report.get("privacy_policy")
    if not isinstance(privacy_policy, dict):
        privacy_policy = {"url": "", "found": False, "applicable": False}
        report["privacy_policy"] = privacy_policy

    if not isinstance(privacy_policy.get("url"), (str, type(None))):
        privacy_policy["url"] = ""
    if not isinstance(privacy_policy.get("found"), bool):
        privacy_policy["found"] = False
    if not isinstance(privacy_policy.get("applicable"), bool):
        privacy_policy["applicable"] = False

    visit = report.get("visit")
    if not isinstance(visit, dict):
        visit = {"observed_at": None, "duration_seconds": 30}
        report["visit"] = visit

    if not isinstance(visit.get("observed_at"), (str, type(None))):
        visit["observed_at"] = None
    if not is_integer(visit.get("duration_seconds")):
        visit["duration_seconds"] = 30

    analysis = report.get("analysis")
    if not isinstance(analysis, dict):
        analysis = {}
        report["analysis"] = analysis

    findings = report.get("findings")
    if not isinstance(findings, list):
        raise ValueError("report.findings must be an array")

    policy_applicable = bool(privacy_policy.get("found") and privacy_policy.get("applicable"))
    normalized_findings = []

    for index, source_finding in enumerate(findings):
        if not isinstance(source_finding, dict):
            continue

        finding = deep_copy(source_finding)
        behavior = finding.get("behavior")
        category = finding.get("category")

        if not isinstance(behavior, str) or not behavior:
            behavior = f"behavior_{index + 1}"
        if not isinstance(category, str) or not category:
            category = "other"

        finding["behavior"] = behavior
        finding["category"] = category

        if not isinstance(finding.get("description"), str):
            finding["description"] = f"Veilance observed activity associated with {behavior}."
        if not isinstance(finding.get("explanation"), str):
            finding["explanation"] = "The finding is based only on the supplied telemetry and policy evidence."

        policy = finding.get("policy")
        if not isinstance(policy, dict):
            policy = {"status": "indeterminate", "evidence": "", "section": ""}
            finding["policy"] = policy

        policy["status"] = normalize_policy_status(policy.get("status"))
        if not isinstance(policy.get("evidence"), str):
            policy["evidence"] = ""
        if not isinstance(policy.get("section"), str):
            policy["section"] = ""

        telemetry = finding.get("telemetry")
        if not isinstance(telemetry, dict):
            telemetry = {"status": "observed", "evidence": [], "observation_count": 1}
            finding["telemetry"] = telemetry

        telemetry_status = telemetry.get("status")
        if telemetry_status not in ALLOWED_TELEMETRY_STATUSES:
            telemetry_status = "observed"
        telemetry["status"] = telemetry_status

        evidence = telemetry.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        telemetry["evidence"] = [item for item in evidence if isinstance(item, str)]

        observation_count = telemetry.get("observation_count")
        if not is_integer(observation_count):
            observation_count = 0 if telemetry_status == "not_observed" else max(1, len(telemetry["evidence"]))
        telemetry["observation_count"] = max(0, observation_count)

        finding["comparison"] = normalize_comparison(finding.get("comparison"), policy_applicable)
        finding["severity"] = normalize_severity(finding.get("severity"))
        finding["confidence"] = normalize_confidence(finding.get("confidence"))

        if not policy_applicable:
            finding["comparison"] = "indeterminate"
            finding["policy"]["status"] = "indeterminate"
            finding["policy"]["evidence"] = ""
            finding["policy"]["section"] = ""

        normalized_findings.append(finding)

    report["findings"] = normalized_findings

    if not isinstance(report.get("important_limitations"), list):
        report["important_limitations"] = []
    report["important_limitations"] = [
        item for item in report["important_limitations"] if isinstance(item, str)
    ]

    analysis["overall_confidence"] = normalize_confidence(analysis.get("overall_confidence"))
    recompute_report(report)
    return report


# ============================================================
# Counterfactual report mutation
# ============================================================

def policy_template(config: dict, comparison: str, category: str, rng: random.Random) -> str:
    templates = config.get("synthetic_policy_templates", {}).get(comparison, [])
    templates = [item for item in templates if isinstance(item, str)]
    if not templates:
        return ""
    return rng.choice(templates).replace("{category}", category)


def explanation_template(config: dict, comparison: str, category: str, rng: random.Random) -> str:
    templates = config.get("explanation_templates", {}).get(comparison, [])
    templates = [item for item in templates if isinstance(item, str)]
    if not templates:
        return "The comparison is based on the supplied telemetry and applicable policy evidence."
    return rng.choice(templates).replace("{category}", category)


def allowed_transition_targets(finding: dict, policy_applicable: bool) -> List[str]:
    current = finding.get("comparison")
    telemetry_status = finding.get("telemetry", {}).get("status")

    if not policy_applicable:
        return ["indeterminate"]

    # Do not invent observed telemetry for a policy-only source finding.
    if telemetry_status == "not_observed" or current == "policy_only":
        return ["policy_only"]

    # Keep observed telemetry observed. Only the policy relationship changes.
    return [
        "matched",
        "partially_matched",
        "observed_only",
        "possible_contradiction",
    ]


def policy_status_for_comparison(config: dict, comparison: str) -> str:
    configured = config.get("comparisons", {}).get("policy_status_by_comparison", {}).get(comparison)
    if configured in ALLOWED_POLICY_STATUSES:
        return configured

    defaults = {
        "matched": "explicitly_disclosed",
        "partially_matched": "broadly_disclosed",
        "policy_only": "explicitly_disclosed",
        "observed_only": "not_clearly_disclosed",
        "possible_contradiction": "contradicted",
        "indeterminate": "indeterminate",
    }
    return defaults[comparison]


def severity_for_comparison(config: dict, comparison: str, current: str) -> str:
    configured = config.get("comparisons", {}).get("severity_by_comparison", {}).get(comparison)
    if configured in ALLOWED_SEVERITIES:
        return configured
    if current in ALLOWED_SEVERITIES:
        return current
    return "low"


def mutate_report_semantics(report: dict, config: dict, rng: random.Random) -> dict:
    output = deep_copy(report)
    privacy_policy = output["privacy_policy"]
    policy_applicable = bool(privacy_policy["found"] and privacy_policy["applicable"])

    # Occasionally turn an applicable-policy example into a whole-report
    # no-applicable-policy counterfactual. This keeps the report internally consistent.
    no_policy_probability = config.get("generation", {}).get("no_policy_probability", 0.08)
    make_no_policy = policy_applicable and is_number(no_policy_probability) and rng.random() < no_policy_probability

    if make_no_policy:
        privacy_policy["found"] = False
        privacy_policy["applicable"] = False
        privacy_policy["url"] = ""
        policy_applicable = False

    counterfactual_probability = config.get("comparisons", {}).get("counterfactual_probability", 0.35)
    if not is_number(counterfactual_probability):
        counterfactual_probability = 0.35

    for finding in output["findings"]:
        category = finding["category"]
        current = finding["comparison"]

        targets = allowed_transition_targets(finding, policy_applicable)
        target = current

        if not policy_applicable:
            target = "indeterminate"
        elif rng.random() < counterfactual_probability:
            candidates = [candidate for candidate in targets if candidate != current]
            if candidates:
                target = rng.choice(candidates)

        finding["comparison"] = target
        finding["policy"]["status"] = policy_status_for_comparison(config, target)
        finding["severity"] = severity_for_comparison(config, target, finding["severity"])
        finding["explanation"] = explanation_template(config, target, category, rng)

        if target == "indeterminate":
            finding["policy"]["evidence"] = ""
            finding["policy"]["section"] = ""
        else:
            text = policy_template(config, target, category, rng)
            if text:
                finding["policy"]["evidence"] = text
                finding["policy"]["section"] = "Synthetic Privacy Statement"

        if target == "policy_only":
            finding["telemetry"]["status"] = "not_observed"
            finding["telemetry"]["evidence"] = []
            finding["telemetry"]["observation_count"] = 0
        else:
            finding["telemetry"]["status"] = "observed"

    apply_report_mutations(output, config.get("mutation_rules", {}), rng)
    recompute_report(output)
    return output


def recompute_counts(findings: list) -> dict:
    counts = {field: 0 for field in COUNT_FIELDS.values()}
    for finding in findings:
        comparison = finding.get("comparison")
        field = COUNT_FIELDS.get(comparison)
        if field:
            counts[field] += 1
    return counts


def recompute_report(report: dict) -> None:
    findings = report["findings"]
    report["analysis"]["counts"] = recompute_counts(findings)

    confidences = [
        finding["confidence"]
        for finding in findings
        if is_number(finding.get("confidence"))
    ]
    if confidences:
        report["analysis"]["overall_confidence"] = round(sum(confidences) / len(confidences), 2)
    else:
        report["analysis"]["overall_confidence"] = 0.75

    counts = report["analysis"]["counts"]
    parts = [f"{value} {key.replace('_', ' ')}" for key, value in counts.items() if value]
    report["analysis"]["summary"] = (
        "Synthetic Veilance analysis identified " + ", ".join(parts) + "."
        if parts
        else "Synthetic Veilance analysis produced no comparison findings."
    )


# ============================================================
# Synthetic telemetry construction
# ============================================================

def extract_api_names(evidence: List[str]) -> List[Tuple[str, str]]:
    result = []
    seen = set()
    for line in evidence:
        match = API_EVIDENCE_RE.search(line.strip())
        if not match:
            continue
        pair = (match.group(1), match.group(2))
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def extract_hosts(evidence: List[str]) -> List[str]:
    hosts = []
    seen = set()
    for line in evidence:
        for match in HOST_RE.finditer(line):
            host = match.group(1).lower()
            # Ignore synthetic/local labels that are clearly API names rather than hosts.
            if host not in seen and "." in host:
                seen.add(host)
                hosts.append(host)
    return hosts


def default_api_names(category: str) -> List[Tuple[str, str]]:
    mapping = {
        "cookies": [("Cookie", "read")],
        "browser_storage": [("Storage", "write")],
        "browser_characteristics": [("Navigator", "read-user-agent"), ("Navigator", "read-language")],
        "device_characteristics": [("Navigator", "read-user-agent"), ("Screen", "read-width")],
        "network_characteristics": [("NetworkInformation", "read-effective-type")],
        "fingerprinting": [("WebGL", "renderer-query"), ("Canvas", "readback")],
        "permissions": [("Permissions", "query")],
        "webrtc": [("WebRTC", "create-offer")],
        "analytics": [("Performance", "get-entries-by-type")],
    }
    return mapping.get(category, [])


def is_third_party_category(category: str) -> bool:
    return category in {
        "third_party_requests",
        "tracking",
        "analytics",
        "advertising",
        "social_media_tracking",
    }


def distribute_count(total: int, item_count: int) -> List[int]:
    if item_count <= 0:
        return []
    total = max(total, item_count)
    base = total // item_count
    remainder = total % item_count
    return [base + (1 if index < remainder else 0) for index in range(item_count)]


def synthesize_finding_telemetry(
    finding: dict,
    domain: str,
    variant_token: str,
    rng: random.Random,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    telemetry = finding["telemetry"]
    category = finding["category"]
    behavior = finding["behavior"]

    if telemetry["status"] == "not_observed" or finding["comparison"] == "policy_only":
        telemetry["status"] = "not_observed"
        telemetry["observation_count"] = 0
        telemetry["evidence"] = [
            f"No corresponding {category} activity was observed during this synthetic visit."
        ]
        return [], [], [], {}

    source_evidence = telemetry.get("evidence", [])
    api_names = extract_api_names(source_evidence)
    hosts = extract_hosts(source_evidence)

    if not api_names:
        api_names = default_api_names(category)

    source_count = telemetry.get("observation_count")
    if not is_integer(source_count) or source_count < 1:
        source_count = max(1, len(api_names) + len(hosts))

    # Jitter the generated count, but never parse/count-cast from prose.
    delta = max(1, round(source_count * 0.30))
    generated_count = rng.randint(max(1, source_count - delta), source_count + delta)

    signals = []
    host_records = []
    trackers = []
    new_evidence = []

    signal_counts = distribute_count(generated_count, len(api_names))
    for index, ((api, action), count) in enumerate(zip(api_names, signal_counts)):
        signals.append({
            "indicatorId": f"synthetic-{stable_hash(f'{variant_token}:{api}:{action}:{index}')}",
            "api": api,
            "action": action,
            "count": count,
        })
        suffix = "time" if count == 1 else "times"
        new_evidence.append(f"{api}.{action} observed {count} {suffix}")

    if is_third_party_category(category) and not hosts:
        hosts = [f"service-{stable_hash(f'{variant_token}:{category}')[:8]}.example"]

    host_request_total = 0
    for index, host in enumerate(hosts[:4]):
        requests = rng.randint(1, max(1, min(12, generated_count)))
        host_request_total += requests
        host_records.append({
            "host": host,
            "requests": requests,
            "types": {"script": requests},
        })
        suffix = "request" if requests == 1 else "requests"
        new_evidence.append(f"Communication with {host} observed for {requests} {suffix}")

    if category in {"analytics", "advertising", "tracking", "social_media_tracking"}:
        tracker_requests = max(1, host_request_total or min(generated_count, 8))
        trackers.append({
            "id": f"synthetic-{stable_hash(f'{variant_token}:{category}:tracker')}",
            "category": category,
            "requests": tracker_requests,
        })
        suffix = "request" if tracker_requests == 1 else "requests"
        new_evidence.append(
            f"Veilance {category} tracker classification observed for {tracker_requests} {suffix}"
        )

    page_updates = {}
    if category == "cookies":
        page_updates["accessibleCookieCount"] = rng.randint(0, 20)
        new_evidence.append(
            f"accessibleCookieCount was {page_updates['accessibleCookieCount']} at the time of observation"
        )
    elif category == "browser_storage":
        page_updates["localStorageKeyCount"] = rng.randint(0, 40)
        page_updates["sessionStorageKeyCount"] = rng.randint(0, 10)
        page_updates["indexedDbCount"] = rng.randint(0, 5)
        page_updates["cacheCount"] = rng.randint(0, 5)
        new_evidence.extend([
            f"localStorageKeyCount was {page_updates['localStorageKeyCount']}",
            f"sessionStorageKeyCount was {page_updates['sessionStorageKeyCount']}",
            f"indexedDbCount was {page_updates['indexedDbCount']}",
            f"cacheCount was {page_updates['cacheCount']}",
        ])

    observation_count = 0
    if signals:
        observation_count += sum(item["count"] for item in signals)
    if host_records and not signals:
        observation_count += sum(item["requests"] for item in host_records)
    if trackers and not signals and not host_records:
        observation_count += sum(item["requests"] for item in trackers)
    if observation_count < 1:
        observation_count = generated_count

    telemetry["status"] = "observed"
    telemetry["observation_count"] = observation_count
    telemetry["evidence"] = new_evidence or [
        f"Synthetic telemetry observed activity associated with {behavior}."
    ]

    return signals, host_records, trackers, page_updates


def build_synthetic_snapshot(report: dict, source_id: str, variant_index: int, seed: str, rng: random.Random) -> dict:
    domain = report["domain"]
    hostname = host_from_url(domain)
    token = stable_hash(f"{source_id}:{variant_index}:{seed}")

    all_signals = []
    all_hosts = []
    all_trackers = []
    page = {
        "scriptCount": 0,
        "thirdPartyScriptCount": 0,
        "iframeCount": 0,
        "thirdPartyIframeCount": 0,
        "accessibleCookieCount": 0,
        "localStorageKeyCount": 0,
        "sessionStorageKeyCount": 0,
        "indexedDbCount": 0,
        "cacheCount": 0,
        "serviceWorker": False,
    }

    for finding in report["findings"]:
        signals, hosts, trackers, page_updates = synthesize_finding_telemetry(
            finding,
            domain,
            token,
            rng,
        )
        all_signals.extend(signals)
        all_hosts.extend(hosts)
        all_trackers.extend(trackers)
        page.update(page_updates)

    # Merge duplicate signals.
    merged_signals = {}
    for signal in all_signals:
        key = (signal["api"], signal["action"])
        if key not in merged_signals:
            merged_signals[key] = deep_copy(signal)
        else:
            merged_signals[key]["count"] += signal["count"]
    signals = list(merged_signals.values())

    # Merge duplicate hosts.
    merged_hosts = {}
    for host in all_hosts:
        key = host["host"]
        if key not in merged_hosts:
            merged_hosts[key] = deep_copy(host)
        else:
            merged_hosts[key]["requests"] += host["requests"]
            merged_hosts[key]["types"]["script"] += host["types"]["script"]
    third_party_hosts = list(merged_hosts.values())

    # Merge tracker categories.
    merged_trackers = {}
    for tracker in all_trackers:
        key = tracker["category"]
        if key not in merged_trackers:
            merged_trackers[key] = deep_copy(tracker)
        else:
            merged_trackers[key]["requests"] += tracker["requests"]
    trackers = list(merged_trackers.values())

    third_party_requests = sum(item["requests"] for item in third_party_hosts)
    first_party_requests = rng.randint(1, max(2, 8 + len(signals) * 2))
    total_requests = first_party_requests + third_party_requests

    page["thirdPartyScriptCount"] = min(third_party_requests, rng.randint(0, max(0, third_party_requests)))
    page["scriptCount"] = page["thirdPartyScriptCount"] + rng.randint(0, max(1, first_party_requests))
    page["iframeCount"] = rng.randint(0, 3)
    page["thirdPartyIframeCount"] = rng.randint(0, page["iframeCount"])
    page["serviceWorker"] = rng.choice([False, False, False, True])

    duration = report["visit"]["duration_seconds"]

    return {
        "schemaVersion": SNAPSHOT_SCHEMA,
        "eventId": f"synthetic-{token}",
        "extensionVersion": "synthetic-training",
        "site": {
            "hostname": hostname,
            "https": domain.startswith("https://"),
        },
        "observation": {
            "observedAt": report["visit"]["observed_at"],
            "durationSeconds": duration,
            "totalRequests": total_requests,
            "firstPartyRequests": first_party_requests,
            "thirdPartyRequests": third_party_requests,
        },
        "thirdPartyHosts": third_party_hosts,
        "trackers": trackers,
        "signals": signals,
        "page": page,
        "security": {
            "headersObserved": False,
        },
        "interest": {
            "score": min(100, len(signals) * 4 + len(trackers) * 8 + len(third_party_hosts) * 3),
        },
        "redactedDocument": {},
    }


# ============================================================
# Policy-document construction
# ============================================================

def build_policy_document(report: dict) -> dict:
    policy = report["privacy_policy"]
    found = policy["found"]
    applicable = policy["applicable"]

    sections = []
    seen = set()

    if found and applicable:
        for finding in report["findings"]:
            evidence = finding["policy"]["evidence"]
            section = finding["policy"]["section"]
            if not evidence:
                continue
            key = (section, evidence)
            if key in seen:
                continue
            seen.add(key)
            sections.append({
                "heading": section or "Privacy Policy",
                "text": evidence,
            })

    return {
        "url": policy["url"],
        "found": found,
        "applicable": applicable,
        "complete": True,
        "sections": sections,
    }


# ============================================================
# Final expected-report cleanup
# ============================================================

def genericize_report_text(report: dict, config: dict, rng: random.Random) -> None:
    # Avoid carrying source-company/provider names into synthetic examples.
    for finding in report["findings"]:
        behavior = finding["behavior"]
        category = finding["category"]
        comparison = finding["comparison"]

        if finding["telemetry"]["status"] == "observed":
            finding["description"] = f"Veilance observed activity associated with {behavior} during the synthetic visit."
        else:
            finding["description"] = f"The policy describes {behavior}, but corresponding activity was not observed during the synthetic visit."

        finding["explanation"] = explanation_template(config, comparison, category, rng)

    duration = report["visit"]["duration_seconds"]
    report["important_limitations"] = [
        f"This telemetry represents a {duration}-second synthetic browser observation. Absence of a behavior in this sample does not demonstrate that the behavior never occurs.",
        "Veilance observed network destinations and browser activity but did not inspect request or response contents. Third-party communication does not by itself establish what personal information was transmitted.",
        "Browser API access does not necessarily establish the purpose for which the accessed information was used.",
        "Existing cookies, browser storage entries, IndexedDB databases, cache entries, or service-worker state may have existed before the observation.",
        "Tracker classifications identify Veilance classifications and do not by themselves establish request contents, profiling, targeted advertising, or data sale.",
    ]


# ============================================================
# Variant count
# ============================================================

def generation_cap(source_count: int, config: dict) -> int:
    caps = config.get("generation", {}).get("source_count_caps", {})
    if source_count <= 1:
        return caps.get("single", 500)
    if source_count <= 10:
        return caps.get("up_to_10", 400)
    if source_count <= 25:
        return caps.get("up_to_25", 300)
    if source_count <= 100:
        return caps.get("up_to_100", 200)
    return caps.get("over_100", 150)


def automatic_variant_count(record: dict, mode: str, source_count: int, config: dict) -> int:
    minimum = config.get("generation", {}).get("minimum_per_source", 75)
    maximum = generation_cap(source_count, config)
    report = normalize_report(get_source_report(record, mode))
    findings = report["findings"]

    categories = {finding["category"] for finding in findings}
    comparisons = {finding["comparison"] for finding in findings}
    policy_evidence_count = sum(1 for finding in findings if finding["policy"]["evidence"])
    telemetry_evidence_count = sum(len(finding["telemetry"]["evidence"]) for finding in findings)

    target = (
        50
        + len(findings) * 50
        + len(categories) * 20
        + len(comparisons) * 15
        + policy_evidence_count * 5
        + min(telemetry_evidence_count, 20)
    )
    return max(minimum, min(target, maximum))


# ============================================================
# Validation of raw training rows
# ============================================================

def validate_snapshot_shape(snapshot: dict) -> None:
    expected_keys = {
        "schemaVersion", "eventId", "extensionVersion", "site", "observation",
        "thirdPartyHosts", "trackers", "signals", "page", "security", "interest",
        "redactedDocument",
    }
    if not isinstance(snapshot, dict):
        raise ValueError("telemetry must be an object")
    if snapshot.get("schemaVersion") != SNAPSHOT_SCHEMA:
        raise ValueError("telemetry schemaVersion mismatch")
    missing = expected_keys - set(snapshot)
    extra = set(snapshot) - expected_keys
    if missing:
        raise ValueError(f"telemetry missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"telemetry has unexpected keys: {sorted(extra)}")
    if not isinstance(snapshot.get("signals"), list):
        raise ValueError("telemetry.signals must be an array")
    if not isinstance(snapshot.get("thirdPartyHosts"), list):
        raise ValueError("telemetry.thirdPartyHosts must be an array")
    if not isinstance(snapshot.get("trackers"), list):
        raise ValueError("telemetry.trackers must be an array")


def validate_expected_report(report: dict) -> None:
    required = {"domain", "privacy_policy", "visit", "analysis", "findings", "important_limitations"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"expected report missing keys: {sorted(missing)}")

    if not isinstance(report["domain"], str) or not report["domain"]:
        raise ValueError("expected.domain must be a non-empty string")

    policy = report["privacy_policy"]
    if not isinstance(policy, dict):
        raise ValueError("expected.privacy_policy must be an object")
    if not isinstance(policy.get("found"), bool) or not isinstance(policy.get("applicable"), bool):
        raise ValueError("expected.privacy_policy found/applicable must be boolean")
    if not isinstance(policy.get("url"), (str, type(None))):
        raise ValueError("expected.privacy_policy.url must be string or null")

    visit = report["visit"]
    if not isinstance(visit, dict) or not is_integer(visit.get("duration_seconds")):
        raise ValueError("expected.visit.duration_seconds must be an integer")

    findings = report["findings"]
    if not isinstance(findings, list):
        raise ValueError("expected.findings must be an array")

    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"expected.findings[{index}] must be an object")
        if finding.get("comparison") not in ALLOWED_COMPARISONS:
            raise ValueError(f"expected.findings[{index}].comparison invalid")
        if finding.get("severity") not in ALLOWED_SEVERITIES:
            raise ValueError(f"expected.findings[{index}].severity invalid")
        if finding.get("policy", {}).get("status") not in ALLOWED_POLICY_STATUSES:
            raise ValueError(f"expected.findings[{index}].policy.status invalid")
        if finding.get("telemetry", {}).get("status") not in ALLOWED_TELEMETRY_STATUSES:
            raise ValueError(f"expected.findings[{index}].telemetry.status invalid")
        if not is_integer(finding.get("telemetry", {}).get("observation_count")):
            raise ValueError(f"expected.findings[{index}].telemetry.observation_count must be integer")

    calculated = recompute_counts(findings)
    if report["analysis"].get("counts") != calculated:
        raise ValueError(
            f"expected.analysis.counts mismatch: actual={report['analysis'].get('counts')!r} "
            f"calculated={calculated!r}"
        )


def validate_training_row(row: dict) -> None:
    if not isinstance(row.get("telemetry"), dict):
        raise ValueError("missing telemetry")
    if not isinstance(row.get("policy_document"), dict):
        raise ValueError("missing policy_document")
    if not isinstance(row.get("expected"), dict):
        raise ValueError("missing expected")

    validate_snapshot_shape(row["telemetry"])
    validate_expected_report(row["expected"])

    hostname = row["telemetry"]["site"]["hostname"]
    if hostname not in row["expected"]["domain"]:
        raise ValueError(
            f"expected.domain does not contain telemetry hostname {hostname}"
        )

    policy_document = row["policy_document"]
    for key in ["url", "found", "applicable", "sections"]:
        if key not in policy_document:
            raise ValueError(f"policy_document missing {key}")
    if not isinstance(policy_document["sections"], list):
        raise ValueError("policy_document.sections must be an array")


# ============================================================
# Generate one trainer-ready raw row
# ============================================================

def generate_variant(
    source: dict,
    mode: str,
    source_id: str,
    variant_index: int,
    seed: str,
    config: dict,
    mutation_rules: dict,
    include_metadata: bool,
) -> dict:
    variant_seed = f"{seed}:{source_id}:{variant_index}"
    rng = random.Random(variant_seed)

    source_report = normalize_report(get_source_report(source, mode))
    source_domain = source_report["domain"]
    new_domain = synthetic_domain(source_domain, variant_index, seed)

    report = deep_copy(source_report)
    report["domain"] = new_domain

    if report["privacy_policy"]["found"] and report["privacy_policy"]["applicable"]:
        report["privacy_policy"]["url"] = synthetic_policy_url(new_domain)

    # Apply semantic mutations first.
    local_config = deep_copy(config)
    local_config["mutation_rules"] = mutation_rules
    report = mutate_report_semantics(report, local_config, rng)

    # Raw telemetry is generated from the mutated expected report, then the
    # expected finding evidence is rewritten from the exact synthetic raw data.
    telemetry = build_synthetic_snapshot(
        report,
        source_id,
        variant_index,
        seed,
        rng,
    )

    # Now that finding evidence has been synchronized with the generated raw
    # telemetry, finalize generic descriptions/limitations and counts.
    genericize_report_text(report, config, rng)
    recompute_report(report)

    policy_document = build_policy_document(report)

    row = {
        "telemetry": telemetry,
        "policy_document": policy_document,
        "expected": report,
    }

    if include_metadata:
        row["_synthetic_metadata"] = {
            "synthetic": True,
            "source_id": source_id,
            "family_id": source_id,
            "variant_index": variant_index,
            "seed": variant_seed,
            "source_mode": mode,
            "source_domain": source_domain,
            "synthetic_domain": new_domain,
        }

    validate_training_row(row)
    return row


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate trainer-ready synthetic Veilance rows from standalone reports, "
            "database records, or existing training rows. Output is always raw training "
            "JSONL with telemetry, policy_document, and expected."
        )
    )
    parser.add_argument("input", help="Input .json, .jsonl, or directory")
    parser.add_argument("-o", "--output", required=True, help="Output raw training JSONL")
    parser.add_argument("--config", default="generator_config.json")
    parser.add_argument("--mutation-rules", default="mutation_rules.json")
    parser.add_argument("--seed", default="1337")
    parser.add_argument("--count", type=int, help="Override automatic variants per source")
    parser.add_argument("--strip-metadata", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    mutation_rules = load_json(Path(args.mutation_rules))
    records = load_records(Path(args.input))

    if not records:
        raise ValueError("No JSON records found")
    if args.count is not None and args.count < 1:
        raise ValueError("--count must be >= 1")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    invalid = 0
    failed_sources = 0
    planned = 0
    family_sizes = []

    with output_path.open("w", encoding="utf-8") as out:
        for source_index, record in enumerate(records):
            try:
                mode = detect_record_mode(record)
                source_id = stable_hash(json_dumps_stable(record))
                variant_count = args.count or automatic_variant_count(
                    record,
                    mode,
                    len(records),
                    config,
                )
                planned += variant_count
                family_sizes.append(variant_count)

                source_domain = get_source_report(record, mode).get("domain")
                print(
                    f"[source {source_index}] mode={mode} "
                    f"domain={source_domain!r} variants={variant_count}"
                )

                for variant_index in range(variant_count):
                    try:
                        row = generate_variant(
                            source=record,
                            mode=mode,
                            source_id=source_id,
                            variant_index=variant_index,
                            seed=args.seed,
                            config=config,
                            mutation_rules=mutation_rules,
                            include_metadata=not args.strip_metadata,
                        )
                    except Exception as exc:
                        invalid += 1
                        print(
                            f"[invalid] source={source_index} variant={variant_index}: {exc}",
                            file=sys.stderr,
                        )
                        if args.fail_on_invalid:
                            raise
                        continue

                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

            except Exception as exc:
                failed_sources += 1
                print(f"[error] source={source_index}: {exc}", file=sys.stderr)
                if args.fail_on_invalid:
                    raise

    print()
    print(f"source records:   {len(records)}")
    print(f"planned variants: {planned}")
    if family_sizes:
        print(f"smallest family:  {min(family_sizes)}")
        print(f"largest family:   {max(family_sizes)}")
    print(f"written:          {written}")
    print(f"invalid skipped:  {invalid}")
    print(f"source failures:  {failed_sources}")
    print(f"output:           {output_path}")
    print()
    print("Next step:")
    print(
        f"  python prepare_dataset.py --input {output_path} --output-dir data/processed"
    )


if __name__ == "__main__":
    main()
