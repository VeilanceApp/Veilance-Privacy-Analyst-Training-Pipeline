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


def is_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


def is_float(value: Any) -> bool:
    return isinstance(value, float)


def is_number(value: Any) -> bool:
    return (
        is_integer(value)
        or is_float(value)
    )


def same_type(original: Any, mutated: Any) -> bool:
    return type(original) is type(mutated)


def iter_json_file(path: Path) -> Iterable[dict]:
    data = load_json(path)

    if isinstance(data, dict):
        yield data
        return

    if isinstance(data, list):
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{path}: list element {index} is not an object"
                )

            yield item

        return

    raise ValueError(
        f"{path}: top-level JSON must be an object or array"
    )


def iter_jsonl_file(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"{path}:{line_number}: row must be an object"
                )

            yield item


def collect_input_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() not in {
            ".json",
            ".jsonl",
        }:
            raise ValueError(
                "Input file must be .json or .jsonl"
            )

        return [path]

    if path.is_dir():
        return sorted(
            file
            for file in path.rglob("*")
            if file.is_file()
            and file.suffix.lower() in {
                ".json",
                ".jsonl",
            }
        )

    raise ValueError(
        f"Input path does not exist: {path}"
    )


def load_records(path: Path) -> List[dict]:
    records = []

    for file in collect_input_files(path):
        if file.suffix.lower() == ".jsonl":
            records.extend(
                iter_jsonl_file(file)
            )

        else:
            records.extend(
                iter_json_file(file)
            )

    return records


def split_path(path: str) -> List[str]:
    return path.split(".")


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


def find_matching_nodes(
    root: Any,
    path: str,
) -> List[Tuple[Any, Any, Any]]:
    """
    Supports paths such as:

        telemetry.observation.durationSeconds
        telemetry.signals.*.count
        expected.findings.*.confidence

    Returns:
        (parent, key/index, value)
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
                (
                    parent,
                    parent_key,
                    current,
                )
            )
            return

        token = remaining[0]
        rest = remaining[1:]

        if token == "*":
            if isinstance(current, list):
                for index, item in enumerate(current):
                    walk(
                        item,
                        current,
                        index,
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

        if (
            isinstance(current, dict)
            and token in current
        ):
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


def mutate_integer(
    value: int,
    rule: dict,
    rng: random.Random,
) -> int:

    strategy = rule.get("strategy")

    if strategy == "jitter":
        spread = rule.get(
            "spread",
            0.25,
        )

        if not is_number(spread):
            raise ValueError(
                "Integer jitter spread must be numeric"
            )

        minimum = rule.get("min")
        maximum = rule.get("max")

        delta = max(
            1,
            round(
                abs(value) * spread
            ),
        )

        low = value - delta
        high = value + delta

        if is_integer(minimum):
            low = max(
                low,
                minimum,
            )

        if is_integer(maximum):
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
        values = rule.get(
            "values",
            [],
        )

        compatible = [
            candidate
            for candidate in values
            if type(candidate) is type(value)
        ]

        if compatible:
            return rng.choice(
                compatible
            )

    return value


def mutate_float(
    value: float,
    rule: dict,
    rng: random.Random,
) -> float:

    strategy = rule.get("strategy")

    if strategy == "jitter":
        spread = rule.get(
            "spread",
            0.05,
        )

        if not is_number(spread):
            raise ValueError(
                "Float jitter spread must be numeric"
            )

        result = (
            value
            + rng.uniform(
                -spread,
                spread,
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

        precision = rule.get(
            "precision"
        )

        if is_integer(precision):
            result = round(
                result,
                precision,
            )

        return result

    if strategy == "choice":
        values = rule.get(
            "values",
            [],
        )

        compatible = [
            candidate
            for candidate in values
            if type(candidate) is type(value)
        ]

        if compatible:
            return rng.choice(
                compatible
            )

    return value


def mutate_string(
    value: str,
    rule: dict,
    rng: random.Random,
) -> str:

    strategy = rule.get(
        "strategy"
    )

    if strategy == "choice":
        values = rule.get(
            "values",
            [],
        )

        compatible = [
            candidate
            for candidate in values
            if isinstance(
                candidate,
                str,
            )
        ]

        if compatible:
            return rng.choice(
                compatible
            )

    if strategy == "template":
        templates = rule.get(
            "templates",
            [],
        )

        compatible = [
            template
            for template in templates
            if isinstance(
                template,
                str,
            )
        ]

        if compatible:
            return rng.choice(
                compatible
            ).replace(
                "{original}",
                value,
            )

    return value


def mutate_value(
    value: Any,
    rule: dict,
    rng: random.Random,
) -> Any:

    if value is None:
        return None

    if isinstance(value, bool):
        if rule.get("strategy") == "choice":
            values = rule.get(
                "values",
                [],
            )

            compatible = [
                candidate
                for candidate in values
                if type(candidate) is type(value)
            ]

            if compatible:
                return rng.choice(
                    compatible
                )

        return value

    if is_integer(value):
        return mutate_integer(
            value,
            rule,
            rng,
        )

    if is_float(value):
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

    probability = rule.get(
        "probability",
        1.0,
    )

    if not is_number(probability):
        raise ValueError(
            f"{path}: probability must be numeric"
        )

    changed = 0

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
                f"Mutation changed type at {path}: "
                f"{type(current).__name__} -> "
                f"{type(mutated).__name__}"
            )

        if mutated != current:
            parent[key] = mutated
            changed += 1

    return changed


def apply_mutation_rules(
    row: dict,
    rules: dict,
    rng: random.Random,
) -> int:

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

    changed = 0

    for path, rule in mutations.items():
        if not isinstance(
            rule,
            dict,
        ):
            continue

        changed += apply_mutation_rule(
            row,
            path,
            rule,
            rng,
        )

    return changed


def path_exists(
    obj: Any,
    path: str,
) -> bool:

    sentinel = object()

    return (
        get_path(
            obj,
            path,
            sentinel,
        )
        is not sentinel
    )


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
            path_exists(
                record,
                field,
            )
            for field in required
        ):
            return name

    raise ValueError(
        "Record did not match any configured record shape"
    )


def report_from_record(
    record: dict,
    mode: str,
    config: dict,
) -> Optional[dict]:

    shape = get_path(
        config,
        f"record_shapes.{mode}",
        {},
    )

    kind = shape.get(
        "kind"
    )

    if kind == "standalone_report":
        return record

    if kind == "training_row":
        expected_path = get_path(
            config,
            "paths.expected",
            "expected",
        )

        expected = get_path(
            record,
            expected_path,
        )

        if isinstance(
            expected,
            dict,
        ):
            return expected

    return None


def generation_cap(
    source_count: int,
    config: dict,
) -> int:
    """
    The more independent real source records we have,
    the fewer descendants we need from each family.

    Values may be overridden in generator_config.json.
    """

    generation = config.get(
        "generation",
        {},
    )

    caps = generation.get(
        "source_count_caps",
        {},
    )

    if source_count <= 1:
        return caps.get(
            "single",
            500,
        )

    if source_count <= 10:
        return caps.get(
            "up_to_10",
            400,
        )

    if source_count <= 25:
        return caps.get(
            "up_to_25",
            300,
        )

    if source_count <= 100:
        return caps.get(
            "up_to_100",
            200,
        )

    return caps.get(
        "over_100",
        150,
    )


def automatic_variant_count(
    record: dict,
    mode: str,
    source_count: int,
    config: dict,
) -> int:
    """
    Estimate a useful amount of synthetic data based on
    semantic richness rather than an arbitrary fixed count.

    A report with ~4 meaningful findings normally lands
    around 300-400 variants when it is the only source.
    """

    generation = config.get(
        "generation",
        {},
    )

    minimum = generation.get(
        "minimum_per_source",
        75,
    )

    maximum = generation_cap(
        source_count,
        config,
    )

    report = report_from_record(
        record,
        mode,
        config,
    )

    if not isinstance(
        report,
        dict,
    ):
        return minimum

    findings_path = get_path(
        config,
        "paths.findings",
        "findings",
    )

    findings = get_path(
        report,
        findings_path,
        [],
    )

    if not isinstance(
        findings,
        list,
    ):
        return minimum

    valid_findings = [
        finding
        for finding in findings
        if isinstance(
            finding,
            dict,
        )
    ]

    categories = {
        finding.get("category")
        for finding in valid_findings
        if isinstance(
            finding.get("category"),
            str,
        )
        and finding.get("category")
    }

    comparisons = {
        finding.get("comparison")
        for finding in valid_findings
        if isinstance(
            finding.get("comparison"),
            str,
        )
        and finding.get("comparison")
    }

    policy_evidence_count = 0
    telemetry_evidence_count = 0

    for finding in valid_findings:
        policy = finding.get(
            "policy",
            {},
        )

        if isinstance(
            policy,
            dict,
        ):
            evidence = policy.get(
                "evidence"
            )

            if (
                isinstance(
                    evidence,
                    str,
                )
                and evidence
            ):
                policy_evidence_count += 1

        telemetry = finding.get(
            "telemetry",
            {},
        )

        if isinstance(
            telemetry,
            dict,
        ):
            evidence = telemetry.get(
                "evidence",
                [],
            )

            if isinstance(
                evidence,
                list,
            ):
                telemetry_evidence_count += len(
                    evidence
                )

    target = (
        50
        + len(valid_findings) * 50
        + len(categories) * 20
        + len(comparisons) * 15
        + policy_evidence_count * 5
        + min(
            telemetry_evidence_count,
            20,
        )
    )

    if target < minimum:
        return minimum

    if target > maximum:
        return maximum

    return target


def synthetic_domain(
    source_domain: str,
    variant_index: int,
    seed: str,
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


def policy_text_for_comparison(
    category: str,
    comparison: str,
    config: dict,
    rng: random.Random,
) -> str:

    templates = get_path(
        config,
        "synthetic_policy_templates",
        {},
    )

    choices = templates.get(
        comparison,
        [],
    )

    choices = [
        item
        for item in choices
        if isinstance(
            item,
            str,
        )
    ]

    if not choices:
        return ""

    return rng.choice(
        choices
    ).replace(
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

    if not isinstance(
        allowed,
        list,
    ) or not allowed:
        return output

    probability = comparison_config.get(
        "counterfactual_probability",
        0.35,
    )

    if not is_number(probability):
        raise ValueError(
            "counterfactual_probability must be numeric"
        )

    if rng.random() >= probability:
        return output

    current = output.get(
        "comparison"
    )

    candidates = [
        candidate
        for candidate in allowed
        if isinstance(
            candidate,
            str,
        )
        and candidate != current
    ]

    if not candidates:
        return output

    target = rng.choice(
        candidates
    )

    output["comparison"] = target

    category = output.get(
        "category"
    )

    if not isinstance(
        category,
        str,
    ) or not category:
        category = output.get(
            "behavior"
        )

    if not isinstance(
        category,
        str,
    ) or not category:
        category = "behavior"

    policy = output.get(
        "policy"
    )

    if not isinstance(
        policy,
        dict,
    ):
        return output

    telemetry = output.get(
        "telemetry"
    )

    if not isinstance(
        telemetry,
        dict,
    ):
        return output

    mappings = comparison_config.get(
        "policy_status_by_comparison",
        {},
    )

    status = mappings.get(
        target
    )

    if (
        isinstance(
            status,
            str,
        )
        and isinstance(
            policy.get("status"),
            str,
        )
    ):
        policy["status"] = status

    policy_text = policy_text_for_comparison(
        category,
        target,
        config,
        rng,
    )

    if target == "indeterminate":
        if isinstance(
            policy.get("evidence"),
            str,
        ):
            policy["evidence"] = ""

        if isinstance(
            policy.get("section"),
            str,
        ):
            policy["section"] = ""

    elif policy_text:
        if isinstance(
            policy.get("evidence"),
            str,
        ):
            policy["evidence"] = policy_text

        if isinstance(
            policy.get("section"),
            str,
        ):
            policy["section"] = (
                "Synthetic Privacy Statement"
            )

    if target == "policy_only":
        if isinstance(
            telemetry.get("status"),
            str,
        ):
            telemetry["status"] = (
                "not_observed"
            )

        if isinstance(
            telemetry.get("evidence"),
            list,
        ):
            telemetry["evidence"] = []

        observation_count = telemetry.get(
            "observation_count"
        )

        if is_integer(
            observation_count
        ):
            telemetry[
                "observation_count"
            ] = 0

    else:
        if isinstance(
            telemetry.get("status"),
            str,
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

    if (
        isinstance(
            severity,
            str,
        )
        and isinstance(
            output.get("severity"),
            str,
        )
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
        template
        for template in templates
        if isinstance(
            template,
            str,
        )
    ]

    if (
        templates
        and isinstance(
            output.get("explanation"),
            str,
        )
    ):
        output["explanation"] = (
            rng.choice(
                templates
            ).replace(
                "{category}",
                category,
            )
        )

    return output


def recompute_counts(
    findings: list,
    config: dict,
) -> dict:

    mapping = get_path(
        config,
        "analysis.count_fields",
        {},
    )

    result = {
        output_field: 0
        for output_field
        in mapping.values()
    }

    for finding in findings:
        if not isinstance(
            finding,
            dict,
        ):
            continue

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

    values = [
        finding.get(
            "confidence"
        )
        for finding in findings
        if isinstance(
            finding,
            dict,
        )
        and is_float(
            finding.get(
                "confidence"
            )
        )
    ]

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

    segments = [
        f"{value} {key.replace('_', ' ')}"
        for key, value in counts.items()
        if value
    ]

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


def mutate_report(
    report: dict,
    config: dict,
    rng: random.Random,
) -> dict:

    output = deep_copy(
        report
    )

    findings_path = get_path(
        config,
        "paths.findings",
        "findings",
    )

    findings = get_path(
        output,
        findings_path,
        [],
    )

    if not isinstance(
        findings,
        list,
    ):
        return output

    mutated_findings = []

    for finding in findings:
        if isinstance(
            finding,
            dict,
        ):
            mutated_findings.append(
                mutate_finding_comparison(
                    finding,
                    config,
                    rng,
                )
            )

        else:
            mutated_findings.append(
                finding
            )

    set_path(
        output,
        findings_path,
        mutated_findings,
    )

    counts_path = get_path(
        config,
        "paths.analysis_counts",
        "analysis.counts",
    )

    set_path(
        output,
        counts_path,
        recompute_counts(
            mutated_findings,
            config,
        ),
    )

    summary_path = get_path(
        config,
        "paths.analysis_summary",
        "analysis.summary",
    )

    existing_summary = get_path(
        output,
        summary_path,
    )

    if isinstance(
        existing_summary,
        str,
    ):
        set_path(
            output,
            summary_path,
            build_summary(
                mutated_findings,
                config,
            ),
        )

    confidence_path = get_path(
        config,
        "paths.overall_confidence",
        "analysis.overall_confidence",
    )

    existing_confidence = get_path(
        output,
        confidence_path,
    )

    confidence = recompute_confidence(
        mutated_findings
    )

    if (
        is_float(
            existing_confidence
        )
        and confidence is not None
    ):
        set_path(
            output,
            confidence_path,
            confidence,
        )

    return output


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

    variant_seed = (
        f"{seed}:{source_id}:{variant_index}"
    )

    rng = random.Random(
        variant_seed
    )

    row = deep_copy(
        source
    )

    shape = get_path(
        config,
        f"record_shapes.{mode}",
        {},
    )

    kind = shape.get(
        "kind"
    )

    if kind == "training_row":
        report_path = get_path(
            config,
            "paths.expected",
            "expected",
        )

    elif kind == "standalone_report":
        report_path = None

    else:
        raise ValueError(
            f"Unsupported record kind: {kind}"
        )

    report = (
        get_path(
            row,
            report_path,
        )
        if report_path
        else row
    )

    if not isinstance(
        report,
        dict,
    ):
        raise ValueError(
            "Expected report object"
        )

    domain_path = get_path(
        config,
        "paths.domain",
        "domain",
    )

    source_domain = get_path(
        report,
        domain_path,
    )

    if not isinstance(
        source_domain,
        str,
    ):
        raise ValueError(
            "Report domain must already be a string"
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

    report = (
        get_path(
            row,
            report_path,
        )
        if report_path
        else row
    )

    mutated_report = mutate_report(
        report,
        config,
        rng,
    )

    if report_path:
        set_path(
            row,
            report_path,
            mutated_report,
        )

    else:
        row = mutated_report

    apply_mutation_rules(
        row,
        mutation_rules,
        rng,
    )

    if include_metadata:
        row[
            "_synthetic_metadata"
        ] = {
            "synthetic": True,
            "source_id": source_id,
            "family_id": source_id,
            "variant_index": variant_index,
            "seed": variant_seed,
            "source_domain": source_domain,
            "synthetic_domain": new_domain,
        }

    return row


def validate_required_fields(
    row: dict,
    config: dict,
) -> List[str]:

    errors = []

    required = get_path(
        config,
        "validation.required_fields",
        [],
    )

    for path in required:
        if not path_exists(
            row,
            path,
        ):
            errors.append(
                f"missing required field: {path}"
            )

    return errors


def validate_allowed_values(
    row: dict,
    config: dict,
) -> List[str]:

    errors = []

    allowed_fields = get_path(
        config,
        "validation.allowed_values",
        {},
    )

    for path, allowed in allowed_fields.items():
        if not isinstance(
            allowed,
            list,
        ):
            continue

        for _, _, value in find_matching_nodes(
            row,
            path,
        ):
            if value not in allowed:
                errors.append(
                    f"{path}: invalid value {value!r}"
                )

    return errors


def validate_report_counts(
    report: dict,
    config: dict,
) -> List[str]:

    errors = []

    findings_path = get_path(
        config,
        "paths.findings",
        "findings",
    )

    counts_path = get_path(
        config,
        "paths.analysis_counts",
        "analysis.counts",
    )

    findings = get_path(
        report,
        findings_path,
    )

    actual_counts = get_path(
        report,
        counts_path,
    )

    if not isinstance(
        findings,
        list,
    ):
        return errors

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
    mode: str,
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

    shape = get_path(
        config,
        f"record_shapes.{mode}",
        {},
    )

    if shape.get(
        "kind"
    ) == "training_row":

        report = get_path(
            row,
            get_path(
                config,
                "paths.expected",
                "expected",
            ),
        )

    else:
        report = row

    if isinstance(
        report,
        dict,
    ):
        errors.extend(
            validate_report_counts(
                report,
                config,
            )
        )

    return errors


def make_sft_text(
    row: dict,
    config: dict,
) -> dict:

    telemetry_path = get_path(
        config,
        "paths.telemetry",
        "telemetry",
    )

    policy_path = get_path(
        config,
        "paths.policy_document",
        "policy_document",
    )

    expected_path = get_path(
        config,
        "paths.expected",
        "expected",
    )

    telemetry = get_path(
        row,
        telemetry_path,
    )

    policy_document = get_path(
        row,
        policy_path,
    )

    expected = get_path(
        row,
        expected_path,
    )

    if (
        telemetry is None
        or policy_document is None
        or expected is None
    ):
        raise ValueError(
            "SFT output requires a real training-row input "
            "containing telemetry, policy_document, and expected. "
            "The generator will not fabricate typed telemetry "
            "from analyst-report prose."
        )

    system_prompt = get_path(
        config,
        "sft.system_prompt",
        "",
    )

    user_payload = {
        "telemetry": telemetry,
        "policy_document": policy_document,
    }

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate realistic synthetic Veilance data "
            "from JSON, JSONL, or a directory of JSON files."
        )
    )

    parser.add_argument(
        "input",
        help="Input .json, .jsonl, or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSONL",
    )

    parser.add_argument(
        "--config",
        default="generator_config.json",
    )

    parser.add_argument(
        "--mutation-rules",
        default="mutation_rules.json",
    )

    parser.add_argument(
        "--seed",
        default="1337",
        help=(
            "Deterministic seed. Kept as a string; "
            "it is not cast to another type."
        ),
    )

    parser.add_argument(
        "--strip-metadata",
        action="store_true",
    )

    parser.add_argument(
        "--sft-output",
        help=(
            "Optional trainer-ready JSONL. "
            "Requires training-row inputs."
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

    config = load_json(
        Path(
            args.config
        )
    )

    mutation_rules = load_json(
        Path(
            args.mutation_rules
        )
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

    generation_counts = []

    try:
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

                    source_id = stable_hash(
                        json_dumps_stable(
                            record
                        )
                    )

                    variant_count = automatic_variant_count(
                        record=record,
                        mode=mode,
                        source_count=len(
                            records
                        ),
                        config=config,
                    )

                    generation_counts.append(
                        variant_count
                    )

                    print(
                        f"[source {source_index}] "
                        f"mode={mode} "
                        f"variants={variant_count}"
                    )

                    for variant_index in range(
                        variant_count
                    ):
                        row = generate_variant(
                            source=record,
                            mode=mode,
                            source_id=source_id,
                            variant_index=variant_index,
                            seed=args.seed,
                            config=config,
                            mutation_rules=mutation_rules,
                            include_metadata=(
                                not args.strip_metadata
                            ),
                        )

                        errors = validate_row(
                            row,
                            mode,
                            config,
                        )

                        if errors:
                            invalid += 1

                            print(
                                f"[invalid] source={source_index} "
                                f"variant={variant_index}",
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
                            if get_path(
                                config,
                                f"record_shapes.{mode}.kind",
                            ) != "training_row":
                                raise ValueError(
                                    "--sft-output cannot be used "
                                    "with standalone reports because "
                                    "typed telemetry will not be "
                                    "fabricated from prose."
                                )

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
                        f"[error] source {source_index}: {exc}",
                        file=sys.stderr,
                    )

                    if args.fail_on_invalid:
                        raise

    finally:
        if sft_file:
            sft_file.close()

    print()
    print(
        f"source records:   {len(records)}"
    )

    if generation_counts:
        print(
            f"planned variants: {sum(generation_counts)}"
        )

        print(
            f"smallest family:  {min(generation_counts)}"
        )

        print(
            f"largest family:   {max(generation_counts)}"
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
            f"sft output:       {args.sft_output}"
        )


if __name__ == "__main__":
    main()