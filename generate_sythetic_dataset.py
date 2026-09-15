import argparse
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def json_dumps_stable(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Strict type handling
# ---------------------------------------------------------------------------

def is_strict_int(value: Any) -> bool:
    """
    Python bool is a subclass of int.

    We explicitly reject bool here.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


def is_strict_float(value: Any) -> bool:
    return isinstance(value, float)


def is_number(value: Any) -> bool:
    return (
        is_strict_int(value)
        or is_strict_float(value)
    )


def same_type(original: Any, mutated: Any) -> bool:
    """
    Enforce exact scalar type preservation.
    """

    if original is None:
        return mutated is None

    if isinstance(original, bool):
        return isinstance(mutated, bool)

    if is_strict_int(original):
        return is_strict_int(mutated)

    if is_strict_float(original):
        return is_strict_float(mutated)

    if isinstance(original, str):
        return isinstance(mutated, str)

    if isinstance(original, list):
        return isinstance(mutated, list)

    if isinstance(original, dict):
        return isinstance(mutated, dict)

    return type(original) is type(mutated)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def iter_json_file(path: Path) -> Iterable[dict]:
    data = load_json(path)

    if isinstance(data, dict):
        yield data
        return

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{path}: list element {idx} is not an object"
                )
            yield item
        return

    raise ValueError(
        f"{path}: top-level JSON must be object or array"
    )


def iter_jsonl_file(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"{path}:{line_no}: row must be an object"
                )

            yield item


def collect_input_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(
                "Input file must be .json or .jsonl"
            )

        return [path]

    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".json", ".jsonl"}
        )

        return files

    raise ValueError(
        f"Input path does not exist: {path}"
    )


def load_records(path: Path) -> List[dict]:
    records = []

    for file in collect_input_files(path):
        if file.suffix.lower() == ".jsonl":
            records.extend(iter_jsonl_file(file))
        else:
            records.extend(iter_json_file(file))

    return records


# ---------------------------------------------------------------------------
# Dot-path resolution
# ---------------------------------------------------------------------------

def split_path(path: str) -> List[str]:
    return path.split(".")


def find_matching_nodes(
    root: Any,
    path: str,
) -> List[Tuple[Any, Any, Any]]:
    """
    Resolve paths like:

        telemetry.observation.durationSeconds
        telemetry.signals.*.count
        expected.findings.*.confidence

    Returns tuples:
        (parent, key_or_index, current_value)

    No values are changed here.
    """

    parts = split_path(path)
    results = []

    def walk(
        current: Any,
        parent: Any,
        parent_key: Any,
        remaining: List[str],
    ) -> None:
        if not remaining:
            results.append(
                (parent, parent_key, current)
            )
            return

        token = remaining[0]
        rest = remaining[1:]

        if token == "*":
            if isinstance(current, list):
                for idx, item in enumerate(current):
                    walk(
                        item,
                        current,
                        idx,
                        rest,
                    )

            elif isinstance(current, dict):
                for key, item in current.items():
                    walk(
                        item,
                        current,
                        key,
                        rest,
                    )

            return

        if isinstance(current, dict):
            if token in current:
                walk(
                    current[token],
                    current,
                    token,
                    rest,
                )

    walk(
        root,
        None,
        None,
        parts,
    )

    return results


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def mutate_integer(
    value: int,
    rule: dict,
    rng: random.Random,
) -> int:
    """
    Always returns int.
    Never converts to float.
    """

    strategy = rule.get("strategy")

    if strategy == "jitter":
        spread = rule.get("spread", 0.25)

        if not isinstance(spread, (int, float)):
            raise ValueError(
                "jitter spread must be numeric in config"
            )

        minimum = rule.get("min")
        maximum = rule.get("max")

        delta = max(
            1,
            round(abs(value) * spread),
        )

        low = value - delta
        high = value + delta

        if is_strict_int(minimum):
            low = max(
                low,
                minimum,
            )

        if is_strict_int(maximum):
            high = min(
                high,
                maximum,
            )

        if high < low:
            return value

        return rng.randint(
            low,
            high,
        )

    if strategy == "choice":
        values = rule.get("values", [])

        compatible = [
            v
            for v in values
            if is_strict_int(v)
        ]

        if not compatible:
            return value

        return rng.choice(compatible)

    return value


def mutate_float(
    value: float,
    rule: dict,
    rng: random.Random,
) -> float:
    """
    Always returns float.
    """

    strategy = rule.get("strategy")

    if strategy == "jitter":
        spread = rule.get("spread", 0.05)

        if not isinstance(spread, (int, float)):
            raise ValueError(
                "jitter spread must be numeric in config"
            )

        result = (
            value
            + rng.uniform(
                -float(spread),
                float(spread),
            )
        )

        minimum = rule.get("min")
        maximum = rule.get("max")

        if is_number(minimum):
            result = max(
                result,
                minimum,
            )

        if is_number(maximum):
            result = min(
                result,
                maximum,
            )

        precision = rule.get("precision")

        if is_strict_int(precision):
            result = round(
                result,
                precision,
            )

        return float(result)

    if strategy == "choice":
        values = rule.get("values", [])

        compatible = [
            v
            for v in values
            if is_strict_float(v)
        ]

        if not compatible:
            return value

        return rng.choice(compatible)

    return value


def mutate_string(
    value: str,
    rule: dict,
    rng: random.Random,
) -> str:
    strategy = rule.get("strategy")

    if strategy == "choice":
        values = rule.get("values", [])

        compatible = [
            v
            for v in values
            if isinstance(v, str)
        ]

        if not compatible:
            return value

        return rng.choice(compatible)

    if strategy == "template":
        templates = rule.get("templates", [])

        compatible = [
            v
            for v in templates
            if isinstance(v, str)
        ]

        if not compatible:
            return value

        template = rng.choice(compatible)

        return template.replace(
            "{original}",
            value,
        )

    return value


def mutate_value(
    value: Any,
    rule: dict,
    rng: random.Random,
) -> Any:
    """
    Mutates based entirely on original runtime type.

    No implicit coercion is performed.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        strategy = rule.get("strategy")

        if strategy == "choice":
            values = rule.get("values", [])

            compatible = [
                v
                for v in values
                if isinstance(v, bool)
            ]

            if compatible:
                return rng.choice(compatible)

        return value

    if is_strict_int(value):
        return mutate_integer(
            value,
            rule,
            rng,
        )

    if is_strict_float(value):
        return mutate_float(
            value,
            rule,
            rng,
        )

    if isinstance(value, str):
        return mutate_string(
            value,
            rule,
            rng,
        )

    return value


def apply_mutation_rule(
    root: dict,
    path: str,
    rule: dict,
    rng: random.Random,
) -> int:
    nodes = find_matching_nodes(
        root,
        path,
    )

    changed = 0

    probability = rule.get(
        "probability",
        1.0,
    )

    if not isinstance(
        probability,
        (int, float),
    ):
        raise ValueError(
            f"{path}: probability must be numeric"
        )

    for parent, key, current in nodes:
        if parent is None:
            continue

        if rng.random() > probability:
            continue

        mutated = mutate_value(
            current,
            rule,
            rng,
        )

        if not same_type(
            current,
            mutated,
        ):
            raise TypeError(
                f"Mutation rule changed type at {path}: "
                f"{type(current).__name__} -> "
                f"{type(mutated).__name__}"
            )

        parent[key] = mutated

        if mutated != current:
            changed += 1

    return changed


def apply_mutation_rules(
    row: dict,
    rules: dict,
    rng: random.Random,
) -> int:
    count = 0

    mutations = rules.get(
        "mutations",
        {},
    )

    if not isinstance(
        mutations,
        dict,
    ):
        raise ValueError(
            "mutation_rules.json mutations must be an object"
        )

    for path, rule in mutations.items():
        if not isinstance(rule, dict):
            continue

        count += apply_mutation_rule(
            row,
            path,
            rule,
            rng,
        )

    return count


# ---------------------------------------------------------------------------
# Report shape handling
# ---------------------------------------------------------------------------

def get_path(
    obj: Any,
    path: str,
    default: Any = None,
) -> Any:
    current = obj

    for part in split_path(path):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def set_path(
    obj: dict,
    path: str,
    value: Any,
) -> None:
    parts = split_path(path)

    current = obj

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}

        if not isinstance(
            current[part],
            dict,
        ):
            raise ValueError(
                f"Cannot traverse non-object at {part}"
            )

        current = current[part]

    current[parts[-1]] = value


def detect_record_mode(
    record: dict,
    config: dict,
) -> str:
    shapes = config.get(
        "record_shapes",
        {},
    )

    for name, definition in shapes.items():
        required = definition.get(
            "required_fields",
            [],
        )

        if all(
            get_path(
                record,
                field,
                default=object(),
            )
            is not None
            for field in required
        ):
            missing = [
                field
                for field in required
                if get_path(
                    record,
                    field,
                    default=None,
                )
                is None
            ]

            if not missing:
                return name

    raise ValueError(
        "Record did not match any configured record shape"
    )


# ---------------------------------------------------------------------------
# Synthetic domain handling
# ---------------------------------------------------------------------------

def synthetic_domain(
    source_domain: str,
    variant_index: int,
    seed: int,
) -> str:
    token = stable_hash(
        f"{source_domain}:{variant_index}:{seed}"
    )

    return (
        f"https://site-{token}.example"
    )


def replace_domain_recursive(
    value: Any,
    old_domain: str,
    new_domain: str,
) -> Any:
    """
    Strings remain strings.
    All other types are preserved.
    """

    if isinstance(value, str):
        result = value

        if old_domain:
            result = result.replace(
                old_domain,
                new_domain,
            )

            old_host = re.sub(
                r"^https?://",
                "",
                old_domain,
            ).rstrip("/")

            new_host = re.sub(
                r"^https?://",
                "",
                new_domain,
            ).rstrip("/")

            result = result.replace(
                old_host,
                new_host,
            )

        return result

    if isinstance(value, list):
        return [
            replace_domain_recursive(
                item,
                old_domain,
                new_domain,
            )
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: replace_domain_recursive(
                item,
                old_domain,
                new_domain,
            )
            for key, item in value.items()
        }

    return value


# ---------------------------------------------------------------------------
# Comparison mutation
# ---------------------------------------------------------------------------

def policy_text_for_comparison(
    category: str,
    comparison: str,
    config: dict,
    rng: random.Random,
) -> str:
    templates = get_path(
        config,
        "synthetic_policy_templates",
        default={},
    )

    choices = templates.get(
        comparison,
        [],
    )

    choices = [
        item
        for item in choices
        if isinstance(item, str)
    ]

    if not choices:
        return ""

    template = rng.choice(
        choices
    )

    return template.replace(
        "{category}",
        category,
    )


def mutate_finding_comparison(
    finding: dict,
    config: dict,
    rng: random.Random,
) -> dict:
    output = deep_copy(
        finding
    )

    comparison_config = config.get(
        "comparisons",
        {},
    )

    allowed = comparison_config.get(
        "allowed",
        [],
    )

    if not allowed:
        return output

    probability = comparison_config.get(
        "counterfactual_probability",
        0.35,
    )

    if rng.random() >= probability:
        return output

    target = rng.choice(
        allowed
    )

    current = output.get(
        "comparison"
    )

    if target == current:
        return output

    output["comparison"] = target

    category = output.get(
        "category",
        output.get(
            "behavior",
            "behavior",
        ),
    )

    mappings = comparison_config.get(
        "policy_status_by_comparison",
        {},
    )

    status = mappings.get(
        target
    )

    policy = output.get(
        "policy"
    )

    if not isinstance(
        policy,
        dict,
    ):
        policy = {}
        output["policy"] = policy

    if isinstance(
        status,
        str,
    ):
        policy["status"] = status

    new_policy_text = policy_text_for_comparison(
        category,
        target,
        config,
        rng,
    )

    if target == "indeterminate":
        policy["evidence"] = ""
        policy["section"] = ""

    elif new_policy_text:
        policy["evidence"] = new_policy_text
        policy["section"] = (
            "Synthetic Privacy Statement"
        )

    telemetry = output.get(
        "telemetry"
    )

    if not isinstance(
        telemetry,
        dict,
    ):
        telemetry = {}
        output["telemetry"] = telemetry

    if target == "policy_only":
        telemetry["status"] = (
            "not_observed"
        )

        if (
            "observation_count"
            in telemetry
        ):
            telemetry[
                "observation_count"
            ] = None

        telemetry["evidence"] = [
            (
                f"No {category} activity was "
                f"observed during this synthetic visit."
            )
        ]

    else:
        if (
            "status"
            in telemetry
            and isinstance(
                telemetry["status"],
                str,
            )
        ):
            telemetry["status"] = (
                "observed"
            )

    severity_map = comparison_config.get(
        "severity_by_comparison",
        {},
    )

    severity = severity_map.get(
        target
    )

    if isinstance(
        severity,
        str,
    ):
        output["severity"] = severity

    explanation_templates = config.get(
        "explanation_templates",
        {},
    )

    templates = explanation_templates.get(
        target,
        [],
    )

    templates = [
        t
        for t in templates
        if isinstance(t, str)
    ]

    if templates:
        output["explanation"] = (
            rng.choice(
                templates
            ).replace(
                "{category}",
                category,
            )
        )

    return output


# ---------------------------------------------------------------------------
# Report recalculation
# ---------------------------------------------------------------------------

def recompute_counts(
    findings: list,
    config: dict,
) -> dict:
    mapping = get_path(
        config,
        "analysis.count_fields",
        default={},
    )

    result = {}

    for output_field in mapping.values():
        result[output_field] = 0

    for finding in findings:
        comparison = finding.get(
            "comparison"
        )

        output_field = mapping.get(
            comparison
        )

        if output_field:
            result[output_field] += 1

    return result


def recompute_confidence(
    findings: list,
) -> Optional[float]:
    values = []

    for finding in findings:
        confidence = finding.get(
            "confidence"
        )

        if is_strict_float(
            confidence
        ):
            values.append(
                confidence
            )

    if not values:
        return None

    return round(
        sum(values) / len(values),
        2,
    )


def build_summary(
    findings: list,
    config: dict,
) -> str:
    counts = recompute_counts(
        findings,
        config,
    )

    segments = []

    for key, value in counts.items():
        if value:
            segments.append(
                f"{value} {key.replace('_', ' ')}"
            )

    if not segments:
        return (
            "No comparison findings were generated "
            "for this synthetic observation."
        )

    return (
        "Synthetic Veilance analysis identified "
        + ", ".join(segments)
        + "."
    )


# ---------------------------------------------------------------------------
# Policy reconstruction
# ---------------------------------------------------------------------------

def reconstruct_policy_document(
    report: dict,
    config: dict,
) -> dict:
    paths = config.get(
        "paths",
        {},
    )

    findings_path = paths.get(
        "findings",
        "findings",
    )

    findings = get_path(
        report,
        findings_path,
        default=[],
    )

    sections = []
    seen = set()

    if isinstance(
        findings,
        list,
    ):
        for finding in findings:
            if not isinstance(
                finding,
                dict,
            ):
                continue

            policy = finding.get(
                "policy",
                {},
            )

            if not isinstance(
                policy,
                dict,
            ):
                continue

            evidence = policy.get(
                "evidence",
                "",
            )

            section = policy.get(
                "section",
                "",
            )

            if (
                not isinstance(
                    evidence,
                    str,
                )
                or not evidence
            ):
                continue

            if not isinstance(
                section,
                str,
            ):
                section = ""

            key = (
                section,
                evidence,
            )

            if key in seen:
                continue

            seen.add(key)

            sections.append({
                "heading": (
                    section
                    or "Privacy Policy"
                ),
                "text": evidence,
            })

    privacy_policy_path = paths.get(
        "privacy_policy",
        "privacy_policy",
    )

    privacy_policy = get_path(
        report,
        privacy_policy_path,
        default={},
    )

    if not isinstance(
        privacy_policy,
        dict,
    ):
        privacy_policy = {}

    return {
        "url": privacy_policy.get(
            "url",
            "",
        ),
        "found": privacy_policy.get(
            "found",
            True,
        ),
        "applicable": privacy_policy.get(
            "applicable",
            True,
        ),
        "complete": True,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Standalone report conversion
# ---------------------------------------------------------------------------

API_EVIDENCE_PATTERN = re.compile(
    r"^([A-Za-z0-9_]+)\.([A-Za-z0-9_-]+)"
    r"\s+observed\s+(\d+)\s+time"
)


def signal_from_evidence(
    evidence: str,
) -> Optional[dict]:
    if not isinstance(
        evidence,
        str,
    ):
        return None

    match = API_EVIDENCE_PATTERN.search(
        evidence.strip()
    )

    if not match:
        return None

    api = match.group(1)
    action = match.group(2)
    count_text = match.group(3)

    # This is parsing source text, not coercing
    # an existing JSON field.
    #
    # The evidence string intrinsically encodes an integer.
    count = int(
        count_text
    )

    return {
        "indicatorId": (
            f"synthetic-{api.lower()}-{action}"
        ),
        "api": api,
        "action": action,
        "count": count,
    }


def standalone_report_to_training_row(
    report: dict,
    config: dict,
) -> dict:
    paths = config.get(
        "paths",
        {},
    )

    domain_path = paths.get(
        "domain",
        "domain",
    )

    domain = get_path(
        report,
        domain_path,
        default="https://synthetic.example",
    )

    if not isinstance(
        domain,
        str,
    ):
        raise ValueError(
            "Configured report domain must be a string"
        )

    findings_path = paths.get(
        "findings",
        "findings",
    )

    findings = get_path(
        report,
        findings_path,
        default=[],
    )

    signals = []

    if isinstance(
        findings,
        list,
    ):
        for finding in findings:
            if not isinstance(
                finding,
                dict,
            ):
                continue

            telemetry = finding.get(
                "telemetry",
                {},
            )

            if not isinstance(
                telemetry,
                dict,
            ):
                continue

            evidence_items = telemetry.get(
                "evidence",
                [],
            )

            if not isinstance(
                evidence_items,
                list,
            ):
                continue

            for evidence in evidence_items:
                signal = signal_from_evidence(
                    evidence
                )

                if signal:
                    signals.append(
                        signal
                    )

    visit_path = paths.get(
        "visit",
        "visit",
    )

    visit = get_path(
        report,
        visit_path,
        default={},
    )

    if not isinstance(
        visit,
        dict,
    ):
        visit = {}

    duration = visit.get(
        "duration_seconds"
    )

    telemetry = {
        "schemaVersion": config.get(
            "default_telemetry_schema",
            "veilance.telemetry-snapshot.v2",
        ),
        "eventId": (
            "synthetic-"
            + stable_hash(
                json_dumps_stable(
                    report
                )
            )
        ),
        "extensionVersion": "synthetic",
        "site": {
            "hostname": re.sub(
                r"^https?://",
                "",
                domain,
            ).split("/")[0],
            "https": domain.startswith(
                "https://"
            ),
        },
        "observation": {
            "observedAt": visit.get(
                "observed_at"
            ),
            "durationSeconds": duration,
        },
        "thirdPartyHosts": [],
        "trackers": [],
        "signals": signals,
        "page": {},
        "security": {},
        "interest": {},
        "redactedDocument": {},
    }

    policy_document = (
        reconstruct_policy_document(
            report,
            config,
        )
    )

    return {
        "telemetry": telemetry,
        "policy_document": (
            policy_document
        ),
        "expected": deep_copy(
            report
        ),
    }


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def normalize_source_row(
    record: dict,
    mode: str,
    config: dict,
) -> dict:
    shape = get_path(
        config,
        f"record_shapes.{mode}",
        default={},
    )

    kind = shape.get(
        "kind"
    )

    if kind == "training_row":
        return deep_copy(
            record
        )

    if kind == "standalone_report":
        return (
            standalone_report_to_training_row(
                record,
                config,
            )
        )

    raise ValueError(
        f"Unsupported configured shape kind: {kind}"
    )


def mutate_expected_report(
    expected: dict,
    config: dict,
    rng: random.Random,
) -> dict:
    report = deep_copy(
        expected
    )

    findings_path = get_path(
        config,
        "paths.findings",
        default="findings",
    )

    findings = get_path(
        report,
        findings_path,
        default=[],
    )

    if not isinstance(
        findings,
        list,
    ):
        return report

    mutated_findings = []

    for finding in findings:
        if not isinstance(
            finding,
            dict,
        ):
            mutated_findings.append(
                finding
            )
            continue

        mutated_findings.append(
            mutate_finding_comparison(
                finding,
                config,
                rng,
            )
        )

    set_path(
        report,
        findings_path,
        mutated_findings,
    )

    counts_path = get_path(
        config,
        "paths.analysis_counts",
        default="analysis.counts",
    )

    set_path(
        report,
        counts_path,
        recompute_counts(
            mutated_findings,
            config,
        ),
    )

    summary_path = get_path(
        config,
        "paths.analysis_summary",
        default="analysis.summary",
    )

    set_path(
        report,
        summary_path,
        build_summary(
            mutated_findings,
            config,
        ),
    )

    confidence_path = get_path(
        config,
        "paths.overall_confidence",
        default="analysis.overall_confidence",
    )

    confidence = recompute_confidence(
        mutated_findings
    )

    if confidence is not None:
        existing = get_path(
            report,
            confidence_path,
            default=None,
        )

        if is_strict_float(
            existing
        ):
            set_path(
                report,
                confidence_path,
                confidence,
            )

    return report


def generate_variant(
    source_row: dict,
    source_id: str,
    variant_index: int,
    seed: int,
    config: dict,
    mutation_rules: dict,
    include_metadata: bool,
) -> dict:
    rng = random.Random(
        seed
    )

    row = deep_copy(
        source_row
    )

    paths = config.get(
        "paths",
        {},
    )

    expected_path = paths.get(
        "expected",
        "expected",
    )

    domain_path = paths.get(
        "domain_in_expected",
        "expected.domain",
    )

    source_domain = get_path(
        row,
        domain_path,
        default="https://source.example",
    )

    if not isinstance(
        source_domain,
        str,
    ):
        source_domain = (
            "https://source.example"
        )

    new_domain = synthetic_domain(
        source_domain,
        variant_index,
        seed,
    )

    row = replace_domain_recursive(
        row,
        source_domain,
        new_domain,
    )

    expected = get_path(
        row,
        expected_path,
        default=None,
    )

    if isinstance(
        expected,
        dict,
    ):
        expected = mutate_expected_report(
            expected,
            config,
            rng,
        )

        set_path(
            row,
            expected_path,
            expected,
        )

    # Apply configured scalar mutations last.
    apply_mutation_rules(
        row,
        mutation_rules,
        rng,
    )

    # Rebuild policy input so policy evidence and
    # output evidence stay aligned.
    expected = get_path(
        row,
        expected_path,
        default={},
    )

    if isinstance(
        expected,
        dict,
    ):
        policy_document = (
            reconstruct_policy_document(
                expected,
                config,
            )
        )

        policy_document["url"] = (
            new_domain.rstrip("/")
            + "/privacy"
        )

        set_path(
            row,
            paths.get(
                "policy_document",
                "policy_document",
            ),
            policy_document,
        )

        privacy_policy_path = (
            paths.get(
                "privacy_policy_in_expected",
                "expected.privacy_policy",
            )
        )

        privacy_policy = get_path(
            row,
            privacy_policy_path,
            default=None,
        )

        if isinstance(
            privacy_policy,
            dict,
        ):
            privacy_policy["url"] = (
                policy_document["url"]
            )

    if include_metadata:
        row["_synthetic_metadata"] = {
            "synthetic": True,
            "source_id": source_id,
            "variant_index": (
                variant_index
            ),
            "seed": seed,
            "source_domain": (
                source_domain
            ),
            "synthetic_domain": (
                new_domain
            ),
        }

    return row


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_allowed_values(
    row: dict,
    config: dict,
) -> List[str]:
    errors = []

    validation = config.get(
        "validation",
        {},
    )

    allowed_fields = validation.get(
        "allowed_values",
        {},
    )

    for path, allowed in allowed_fields.items():
        if not isinstance(
            allowed,
            list,
        ):
            continue

        nodes = find_matching_nodes(
            row,
            path,
        )

        for _, _, value in nodes:
            if value not in allowed:
                errors.append(
                    f"{path}: invalid value {value!r}"
                )

    return errors


def validate_required_fields(
    row: dict,
    config: dict,
) -> List[str]:
    errors = []

    required = get_path(
        config,
        "validation.required_fields",
        default=[],
    )

    for path in required:
        sentinel = object()

        value = get_path(
            row,
            path,
            default=sentinel,
        )

        if value is sentinel:
            errors.append(
                f"missing required field: {path}"
            )

    return errors


def validate_count_consistency(
    row: dict,
    config: dict,
) -> List[str]:
    errors = []

    expected_path = get_path(
        config,
        "paths.expected",
        default="expected",
    )

    expected = get_path(
        row,
        expected_path,
        default=None,
    )

    if not isinstance(
        expected,
        dict,
    ):
        return errors

    findings_path = get_path(
        config,
        "paths.findings",
        default="findings",
    )

    findings = get_path(
        expected,
        findings_path,
        default=None,
    )

    if not isinstance(
        findings,
        list,
    ):
        return errors

    actual_counts_path = get_path(
        config,
        "paths.analysis_counts",
        default="analysis.counts",
    )

    actual_counts = get_path(
        expected,
        actual_counts_path,
        default=None,
    )

    calculated = recompute_counts(
        findings,
        config,
    )

    if actual_counts != calculated:
        errors.append(
            "analysis count mismatch: "
            f"actual={actual_counts!r} "
            f"calculated={calculated!r}"
        )

    return errors


def validate_row(
    row: dict,
    config: dict,
) -> List[str]:
    errors = []

    errors.extend(
        validate_required_fields(
            row,
            config,
        )
    )

    errors.extend(
        validate_allowed_values(
            row,
            config,
        )
    )

    errors.extend(
        validate_count_consistency(
            row,
            config,
        )
    )

    return errors


# ---------------------------------------------------------------------------
# Optional SFT conversion
# ---------------------------------------------------------------------------

def make_sft_text(
    row: dict,
    config: dict,
) -> dict:
    sft = config.get(
        "sft",
        {},
    )

    system_prompt = sft.get(
        "system_prompt",
        "",
    )

    telemetry_path = get_path(
        config,
        "paths.telemetry",
        default="telemetry",
    )

    policy_path = get_path(
        config,
        "paths.policy_document",
        default="policy_document",
    )

    expected_path = get_path(
        config,
        "paths.expected",
        default="expected",
    )

    user_payload = {
        "telemetry": get_path(
            row,
            telemetry_path,
        ),
        "policy_document": get_path(
            row,
            policy_path,
        ),
    }

    expected = get_path(
        row,
        expected_path,
    )

    text = (
        "<|im_start|>system\n"
        + system_prompt
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + json.dumps(
            user_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + json.dumps(
            expected,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "<|im_end|>"
    )

    return {
        "text": text
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate configurable synthetic Veilance "
            "training data from JSON, JSONL, or directories."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "Input .json, .jsonl, or directory"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSONL file",
    )

    parser.add_argument(
        "--config",
        default="generator_config.json",
        help=(
            "Generator configuration JSON"
        ),
    )

    parser.add_argument(
        "--mutation-rules",
        default="mutation_rules.json",
        help=(
            "Mutation rules JSON"
        ),
    )

    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=100,
        help=(
            "Synthetic variants per source record"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    parser.add_argument(
        "--strip-metadata",
        action="store_true",
    )

    parser.add_argument(
        "--sft-output",
        help=(
            "Optional second JSONL containing "
            '{"text":"..."} rows for TRL SFTTrainer'
        ),
    )

    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    config_path = Path(
        args.config
    )

    rules_path = Path(
        args.mutation_rules
    )

    if args.count < 1:
        raise ValueError(
            "--count must be >= 1"
        )

    config = load_json(
        config_path
    )

    mutation_rules = load_json(
        rules_path
    )

    records = load_records(
        input_path
    )

    if not records:
        raise ValueError(
            "No JSON records found"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sft_file = None

    if args.sft_output:
        sft_path = Path(
            args.sft_output
        )

        sft_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sft_file = sft_path.open(
            "w",
            encoding="utf-8",
        )

    written = 0
    invalid = 0
    failed_sources = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out:

        for source_index, record in enumerate(
            records
        ):
            try:
                mode = detect_record_mode(
                    record,
                    config,
                )

                source_row = normalize_source_row(
                    record,
                    mode,
                    config,
                )

                source_id = stable_hash(
                    json_dumps_stable(
                        source_row
                    )
                )

                for variant_index in range(
                    args.count
                ):
                    variant_seed = (
                        args.seed
                        + (
                            source_index
                            * 10_000_000
                        )
                        + variant_index
                    )

                    row = generate_variant(
                        source_row=source_row,
                        source_id=source_id,
                        variant_index=(
                            variant_index
                        ),
                        seed=variant_seed,
                        config=config,
                        mutation_rules=(
                            mutation_rules
                        ),
                        include_metadata=(
                            not args.strip_metadata
                        ),
                    )

                    errors = validate_row(
                        row,
                        config,
                    )

                    if errors:
                        invalid += 1

                        print(
                            (
                                f"[invalid] source="
                                f"{source_index} "
                                f"variant="
                                f"{variant_index}"
                            ),
                            file=sys.stderr,
                        )

                        for error in errors:
                            print(
                                f"  - {error}",
                                file=sys.stderr,
                            )

                        if args.fail_on_invalid:
                            raise ValueError(
                                "; ".join(
                                    errors
                                )
                            )

                        continue

                    out.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    if sft_file:
                        sft_row = make_sft_text(
                            row,
                            config,
                        )

                        sft_file.write(
                            json.dumps(
                                sft_row,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    written += 1

            except Exception as exc:
                failed_sources += 1

                print(
                    (
                        f"[error] source "
                        f"{source_index}: {exc}"
                    ),
                    file=sys.stderr,
                )

                if args.fail_on_invalid:
                    raise

    if sft_file:
        sft_file.close()

    print()
    print(
        f"source records:   {len(records)}"
    )
    print(
        f"variants/source:  {args.count}"
    )
    print(
        f"written:          {written}"
    )
    print(
        f"invalid skipped:  {invalid}"
    )
    print(
        f"source failures:  {failed_sources}"
    )
    print(
        f"output:           {output_path}"
    )

    if args.sft_output:
        print(
            f"sft output:       "
            f"{args.sft_output}"
        )


if __name__ == "__main__":
    main()