import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from prompt import SYSTEM_PROMPT, build_user_prompt, compact_json
from schema import validate_report
from telemetry import validate_snapshot_shape


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"{path}:{n}: {e}") from e


def validate_row(row, i):
    for k in ["telemetry", "policy_document", "expected"]:
        if k not in row:
            raise ValueError(f"row {i}: missing {k}")

    validate_snapshot_shape(row["telemetry"])
    validate_report(row["expected"])

    host = row["telemetry"]["site"]["hostname"]
    if host not in row["expected"]["domain"]:
        raise ValueError(
            f"row {i}: expected.domain does not contain telemetry hostname {host}"
        )

    policy = row["policy_document"]
    if not isinstance(policy, dict):
        raise ValueError(f"row {i}: policy_document must be an object")
    for key in ["url", "found", "applicable", "sections"]:
        if key not in policy:
            raise ValueError(f"row {i}: policy_document missing {key}")
    if not isinstance(policy["sections"], list):
        raise ValueError(f"row {i}: policy_document.sections must be a list")


def to_sft(row):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        "completion": [
            {"role": "assistant", "content": compact_json(row["expected"])}
        ],
        "domain": row["telemetry"]["site"]["hostname"],
    }


def family_key(row):
    metadata = row.get("_synthetic_metadata")
    if isinstance(metadata, dict):
        family_id = metadata.get("family_id")
        if isinstance(family_id, str) and family_id:
            return "synthetic-family:" + family_id

    return "domain:" + row["telemetry"]["site"]["hostname"].lower()


def split(rows, seed, tr=0.9, vr=0.05):
    groups = defaultdict(list)
    for row in rows:
        groups[family_key(row)].append(row)

    families = list(groups)
    random.Random(seed).shuffle(families)

    if len(families) < 3:
        raise ValueError(
            "need at least 3 independent source families/domains for train/validation/test splitting; "
            "all descendants of one synthetic source family are intentionally kept together to prevent leakage"
        )

    nt = max(1, int(len(families) * tr))
    nv = max(1, int(len(families) * vr))

    if nt + nv >= len(families):
        nt = len(families) - 2
        nv = 1

    sets = [
        set(families[:nt]),
        set(families[nt:nt + nv]),
        set(families[nt + nv:]),
    ]

    return [[row for family in subset for row in groups[family]] for subset in sets]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_sft(row), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rows = list(read_jsonl(args.input))
    for i, row in enumerate(rows):
        validate_row(row, i)

    train, val, test = split(rows, args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write(out / "train.jsonl", train)
    write(out / "validation.jsonl", val)
    write(out / "test.jsonl", test)

    families = {family_key(row) for row in rows}
    manifest = {
        "rows": len(rows),
        "families": len(families),
        "domains": len({row["telemetry"]["site"]["hostname"] for row in rows}),
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "split_unit": "synthetic family_id when present, otherwise telemetry hostname",
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
