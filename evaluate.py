import argparse
import json

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from infer import generate_report, load_model


def rows(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def expected(row):
    return json.loads(row["completion"][-1]["content"])


def analysis_input(row):
    text = next(
        message["content"] for message in row["prompt"] if message["role"] == "user"
    )
    marker = "INPUT:\n"
    return json.loads(text[text.index(marker) + len(marker) :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=3_200)
    args = parser.parse_args()

    tokenizer, model = load_model(args.base_model, args.adapter)
    expected_comparisons = []
    predicted_comparisons = []
    invalid = 0
    missing = 0
    extra = 0
    false_accusations = 0
    accusations = 0
    policy_status_correct = 0
    telemetry_status_correct = 0
    status_pairs = 0

    for row in tqdm(list(rows(args.test))):
        gold = expected(row)
        try:
            prediction = generate_report(
                tokenizer,
                model,
                analysis_input(row),
                max_new_tokens=args.max_new_tokens,
            )
        except Exception:
            invalid += 1
            continue

        gold_findings = {item["behavior"]: item for item in gold["findings"]}
        predicted_findings = {
            item["behavior"]: item for item in prediction["findings"]
        }
        missing += len(set(gold_findings) - set(predicted_findings))
        extra += len(set(predicted_findings) - set(gold_findings))
        for behavior, gold_finding in gold_findings.items():
            if behavior not in predicted_findings:
                continue
            predicted_finding = predicted_findings[behavior]
            expected_comparisons.append(gold_finding["comparison"])
            predicted_comparisons.append(predicted_finding["comparison"])
            status_pairs += 1
            policy_status_correct += (
                gold_finding["policy"]["status"]
                == predicted_finding["policy"]["status"]
            )
            telemetry_status_correct += (
                gold_finding["telemetry"]["status"]
                == predicted_finding["telemetry"]["status"]
            )
            if predicted_finding["comparison"] in {
                "observed_only",
                "possible_contradiction",
            }:
                accusations += 1
                if gold_finding["comparison"] not in {
                    "observed_only",
                    "possible_contradiction",
                }:
                    false_accusations += 1

    metrics = {
        "invalid_outputs": invalid,
        "matched_findings": len(expected_comparisons),
        "missing_findings": missing,
        "extra_findings": extra,
        "comparison_accuracy": (
            accuracy_score(expected_comparisons, predicted_comparisons)
            if expected_comparisons
            else 0
        ),
        "comparison_macro_f1": (
            f1_score(
                expected_comparisons,
                predicted_comparisons,
                average="macro",
                zero_division=0,
            )
            if expected_comparisons
            else 0
        ),
        "policy_status_accuracy": (
            policy_status_correct / status_pairs if status_pairs else 0
        ),
        "telemetry_status_accuracy": (
            telemetry_status_correct / status_pairs if status_pairs else 0
        ),
        "false_accusations": false_accusations,
        "predicted_accusations": accusations,
        "false_accusation_rate": (
            false_accusations / accusations if accusations else 0
        ),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
