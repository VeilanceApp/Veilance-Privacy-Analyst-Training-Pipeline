import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from generate_synthetic_dataset import (
    generate_variant,
    iter_input,
    load_config,
    source_report,
)
from input_normalization import (
    build_analysis_input,
    normalize_runtime_input,
    validate_policy_document,
)
from prompt import SYSTEM_PROMPT, build_user_prompt, compact_json
from schema import validate_report
from telemetry import validate_snapshot_shape


CONTRACT_VERSION = "veilance.privacy-comparison-training.v2"


def read_jsonl(path):
    """Backward-compatible reader that now accepts JSON, JSONL, and DB dumps."""

    yield from iter_input(Path(path))


def is_training_row(record: dict) -> bool:
    return all(key in record for key in ("telemetry", "policy_document", "expected"))


def expand_input_records(
    records: list[dict],
    *,
    input_format: str,
    synthetic_variants: int,
    generator_seed: str,
    generator_config: dict,
) -> tuple[list[dict], dict]:
    """Convert mixed training rows and processed DB reports to raw training rows."""

    if synthetic_variants < 1:
        raise ValueError("--synthetic-variants must be at least 1")
    rows = []
    direct_rows = 0
    database_sources = 0
    synthetic_rows = 0

    for source_index, record in enumerate(records):
        use_direct = input_format != "database" and is_training_row(record)
        if input_format == "training" and not use_direct:
            raise ValueError(
                f"input record {source_index} is not a raw training row; expected telemetry, policy_document, and expected"
            )
        if use_direct:
            rows.append(record)
            direct_rows += 1
            continue

        try:
            source_report(record)
        except ValueError as exc:
            raise ValueError(
                f"input record {source_index} is neither a training row nor a processed database result: {exc}"
            ) from exc
        database_sources += 1
        for variant_index in range(synthetic_variants):
            rows.append(
                generate_variant(
                    record,
                    source_index,
                    variant_index,
                    generator_seed,
                    generator_config,
                )
            )
            synthetic_rows += 1

    return rows, {
        "input_records": len(records),
        "direct_training_rows": direct_rows,
        "database_source_records": database_sources,
        "synthetic_rows": synthetic_rows,
    }


def _policy_character_count(policy_document: dict) -> int:
    return sum(len(section["text"]) for section in policy_document["sections"])


def _validate_policy_grounding(row: dict, row_index: int) -> None:
    document = row["policy_document"]
    expected = row["expected"]
    expected_policy = expected["privacy_policy"]
    if expected_policy != {
        "url": document["url"] if document["found"] else "",
        "found": document["found"],
        "applicable": document["applicable"],
    }:
        raise ValueError(
            f"row {row_index}: expected.privacy_policy does not match policy_document"
        )

    headings = {
        section["heading"].strip().casefold()
        for section in document["sections"]
        if section["heading"].strip()
    }
    for finding_index, finding in enumerate(expected["findings"]):
        policy = finding["policy"]
        if policy["status"] in {
            "explicitly_disclosed",
            "broadly_disclosed",
            "implicitly_disclosed",
            "contradicted",
        }:
            if not policy["evidence"].strip():
                raise ValueError(
                    f"row {row_index}: finding {finding_index} has a disclosure label without policy evidence"
                )
            referenced = [
                item.strip().casefold()
                for item in policy["section"].split(";")
                if item.strip()
            ]
            if not referenced or any(item not in headings for item in referenced):
                raise ValueError(
                    f"row {row_index}: finding {finding_index} cites a policy section not present in policy_document"
                )


def validate_row(row: dict, row_index: int, max_policy_chars: int) -> dict:
    for key in ("telemetry", "policy_document", "expected"):
        if key not in row:
            raise ValueError(f"row {row_index}: missing {key}")

    snapshot = validate_snapshot_shape(row["telemetry"])
    policy_document = validate_policy_document(row["policy_document"])
    expected = validate_report(row["expected"])

    hostname = snapshot["site"]["hostname"].lower()
    if hostname not in expected["domain"].lower():
        raise ValueError(
            f"row {row_index}: expected.domain does not contain telemetry hostname {hostname}"
        )
    if expected["visit"]["duration_seconds"] != snapshot["observation"]["durationSeconds"]:
        raise ValueError(
            f"row {row_index}: expected.visit.duration_seconds does not match telemetry"
        )
    if _policy_character_count(policy_document) > max_policy_chars:
        raise ValueError(
            f"row {row_index}: policy_document exceeds {max_policy_chars} characters; reduce it to relevant complete sections"
        )
    _validate_policy_grounding(row, row_index)

    runtime = normalize_runtime_input(
        {
            "telemetry": snapshot,
            "domain_url": expected["domain"],
            "privacy_policy_url": policy_document["url"],
            "visit": {
                "observed_at": expected["visit"]["observed_at"],
            },
        }
    )
    return build_analysis_input(runtime, policy_document)


def to_sft(row: dict, analysis_input: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(analysis_input)},
        ],
        "completion": [
            {"role": "assistant", "content": compact_json(row["expected"])}
        ],
        "domain": row["telemetry"]["site"]["hostname"],
        "contract_version": CONTRACT_VERSION,
    }


def family_key(row: dict) -> str:
    metadata = row.get("_synthetic_metadata")
    if isinstance(metadata, dict):
        source_domain = metadata.get("source_domain")
        if isinstance(source_domain, str) and source_domain:
            candidate = (
                source_domain
                if "://" in source_domain
                else "https://" + source_domain
            )
            source_hostname = urlparse(candidate).hostname
            if source_hostname:
                return "domain:" + source_hostname.lower().rstrip(".")
        family_id = metadata.get("family_id")
        if isinstance(family_id, str) and family_id:
            return "synthetic-family:" + family_id
    return "domain:" + row["telemetry"]["site"]["hostname"].lower()


def split(rows: list[dict], seed: int, train_ratio: float, validation_ratio: float):
    groups = defaultdict(list)
    for row in rows:
        groups[family_key(row)].append(row)

    families = list(groups)
    random.Random(seed).shuffle(families)
    if len(families) < 3:
        raise ValueError(
            "need at least 3 independent source families/domains; descendants of one synthetic source remain together to prevent leakage"
        )

    train_count = max(1, int(len(families) * train_ratio))
    validation_count = max(1, int(len(families) * validation_ratio))
    if train_count + validation_count >= len(families):
        train_count = len(families) - 2
        validation_count = 1

    family_sets = [
        families[:train_count],
        families[train_count : train_count + validation_count],
        families[train_count + validation_count :],
    ]
    return [
        [row for family in family_set for row in groups[family]]
        for family_set in family_sets
    ]


def write(path: Path, rows: list[dict], prepared_inputs: dict[int, dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    to_sft(row, prepared_inputs[id(row)]),
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_raw_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate raw Veilance training rows or convert processed database "
            "reports into synthetic rows, then create leakage-safe SFT splits."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--input-format",
        choices=["auto", "training", "database"],
        default="auto",
        help=(
            "auto accepts both raw training rows and processed DB reports; "
            "training requires telemetry/policy_document/expected rows; "
            "database synthesizes every input record from its report"
        ),
    )
    parser.add_argument(
        "--synthetic-variants",
        type=int,
        help="Synthetic rows per processed database record; defaults to generator_config.json.",
    )
    parser.add_argument("--generator-config")
    parser.add_argument(
        "--generator-seed",
        help="Synthetic generation seed; defaults to the split seed.",
    )
    parser.add_argument(
        "--save-raw-rows",
        nargs="?",
        const="raw_training_rows.jsonl",
        help=(
            "Also save the validated raw telemetry/policy/expected rows. With no "
            "path, writes raw_training_rows.jsonl inside --output-dir."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--max-policy-chars", type=int, default=24_000)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1 or not 0 < args.validation_ratio < 1:
        raise ValueError("split ratios must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train-ratio plus validation-ratio must be less than 1")

    source_records = list(iter_input(Path(args.input)))
    if not source_records:
        raise ValueError("input contains no records")
    generator_config = load_config(args.generator_config)
    synthetic_variants = (
        int(generator_config["variants_per_source"])
        if args.synthetic_variants is None
        else args.synthetic_variants
    )
    rows, ingestion = expand_input_records(
        source_records,
        input_format=args.input_format,
        synthetic_variants=synthetic_variants,
        generator_seed=args.generator_seed or str(args.seed),
        generator_config=generator_config,
    )
    prepared_inputs = {
        id(row): validate_row(row, index, args.max_policy_chars)
        for index, row in enumerate(rows)
    }

    train, validation, test = split(
        rows,
        args.seed,
        args.train_ratio,
        args.validation_ratio,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows_path = None
    if args.save_raw_rows:
        raw_rows_path = Path(args.save_raw_rows)
        if not raw_rows_path.is_absolute():
            raw_rows_path = output_dir / raw_rows_path
        write_raw_rows(raw_rows_path, rows)
    write(output_dir / "train.jsonl", train, prepared_inputs)
    write(output_dir / "validation.jsonl", validation, prepared_inputs)
    write(output_dir / "test.jsonl", test, prepared_inputs)

    families = {family_key(row) for row in rows}
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "rows": len(rows),
        "families": len(families),
        "domains": len({row["telemetry"]["site"]["hostname"] for row in rows}),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "split_unit": "original source domain when present, otherwise synthetic family_id or telemetry hostname",
        "max_policy_chars": args.max_policy_chars,
        "input_format": args.input_format,
        **ingestion,
        "synthetic_variants_per_database_record": (
            synthetic_variants if ingestion["database_source_records"] else 0
        ),
        "generator_seed": args.generator_seed or str(args.seed),
        "generator_config_sha256": hashlib.sha256(
            json.dumps(
                generator_config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "raw_rows_output": str(raw_rows_path) if raw_rows_path else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
