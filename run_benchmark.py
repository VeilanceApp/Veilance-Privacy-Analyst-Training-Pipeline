#!/usr/bin/env python3

import argparse
import json
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from infer import (
    configure_logging,
    generate_report,
    load_model,
    prepare_analysis_input,
)


# =============================================================================
# Helpers
# =============================================================================


def load_json(path: str) -> dict:
    value = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"{path} must contain one JSON object"
        )

    return value


def canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_div(a, b):
    return a / b if b else 0.0


def nested_get(obj, *keys, default=None):
    value = obj

    for key in keys:
        if not isinstance(value, dict):
            return default

        value = value.get(
            key,
            default,
        )

    return value


# =============================================================================
# Token correctness
# =============================================================================


def tokenize_json(tokenizer, value):
    return tokenizer.encode(
        canonical_json(value),
        add_special_tokens=False,
    )


def token_sequence_correctness(
    tokenizer,
    expected,
    predicted,
):
    """
    Ordered token similarity.

    This is better than simple position-by-position accuracy because one
    inserted token does not make every token after it appear incorrect.
    """

    gold = tokenize_json(
        tokenizer,
        expected,
    )

    pred = tokenize_json(
        tokenizer,
        predicted,
    )

    if not gold and not pred:
        return 1.0

    if not gold or not pred:
        return 0.0

    return SequenceMatcher(
        None,
        gold,
        pred,
        autojunk=False,
    ).ratio()


def token_overlap(
    tokenizer,
    expected,
    predicted,
):
    """
    Token precision / recall / F1 without considering token order.
    """

    gold = tokenize_json(
        tokenizer,
        expected,
    )

    pred = tokenize_json(
        tokenizer,
        predicted,
    )

    gold_counts = Counter(
        gold
    )

    pred_counts = Counter(
        pred
    )

    overlap = sum(
        (
            gold_counts
            & pred_counts
        ).values()
    )

    precision = safe_div(
        overlap,
        len(pred),
    )

    recall = safe_div(
        overlap,
        len(gold),
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matching_tokens": overlap,
        "expected_tokens": len(gold),
        "predicted_tokens": len(pred),
    }


# =============================================================================
# Finding metrics
# =============================================================================


def findings_by_behavior(report):
    findings = report.get(
        "findings",
        []
    )

    if not isinstance(findings, list):
        return {}

    result = {}

    for finding in findings:
        if not isinstance(
            finding,
            dict,
        ):
            continue

        behavior = finding.get(
            "behavior"
        )

        if isinstance(
            behavior,
            str,
        ):
            result[behavior] = finding

    return result


def behavior_metrics(
    expected_findings,
    predicted_findings,
):
    gold = set(
        expected_findings
    )

    pred = set(
        predicted_findings
    )

    tp = len(
        gold & pred
    )

    fp = len(
        pred - gold
    )

    fn = len(
        gold - pred
    )

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall
        else 0.0
    )

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# Evidence metrics
# =============================================================================


def evidence_accuracy(
    expected_findings,
    predicted_findings,
):
    """
    Checks exact telemetry evidence strings for findings that exist in both
    expected and predicted outputs.
    """

    total = 0
    correct = 0

    for behavior, gold_finding in (
        expected_findings.items()
    ):
        pred_finding = (
            predicted_findings.get(
                behavior
            )
        )

        if pred_finding is None:
            continue

        gold_evidence = nested_get(
            gold_finding,
            "telemetry",
            "evidence",
            default=[],
        )

        pred_evidence = nested_get(
            pred_finding,
            "telemetry",
            "evidence",
            default=[],
        )

        if not isinstance(
            gold_evidence,
            list,
        ):
            gold_evidence = []

        if not isinstance(
            pred_evidence,
            list,
        ):
            pred_evidence = []

        gold_set = set(
            gold_evidence
        )

        pred_set = set(
            pred_evidence
        )

        total += len(
            gold_set | pred_set
        )

        correct += len(
            gold_set & pred_set
        )

    return safe_div(
        correct,
        total,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one Verity model against one "
            "manually verified expected report."
        )
    )

    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-14B",
    )

    parser.add_argument(
        "--adapter",
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Raw Veilance input JSON",
    )

    parser.add_argument(
        "--expected",
        required=True,
        help="Gold/verified Verity report JSON",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3200,
    )

    parser.add_argument(
        "--use-supplied-policy-document",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default="benchmark_results.json",
    )

    parser.add_argument(
        "--prediction-output",
        default="benchmark_prediction.json",
    )

    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
    )

    args = parser.parse_args()

    configure_logging(
        log_level=args.log_level
    )

    # -------------------------------------------------------------------------
    # Load files
    # -------------------------------------------------------------------------

    raw_input = load_json(
        args.input
    )

    expected = load_json(
        args.expected
    )

    print()
    print("=" * 72)
    print("VERITY MODEL BENCHMARK")
    print("=" * 72)
    print(
        f"Base model : {args.base_model}"
    )
    print(
        f"Adapter    : {args.adapter}"
    )
    print(
        f"Input      : {args.input}"
    )
    print(
        f"Expected   : {args.expected}"
    )
    print("=" * 72)
    print()

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    load_started = time.perf_counter()

    tokenizer, model = load_model(
        args.base_model,
        args.adapter,
    )

    model_load_seconds = (
        time.perf_counter()
        - load_started
    )

    # -------------------------------------------------------------------------
    # Prepare exact same analysis input used by normal inference
    # -------------------------------------------------------------------------

    preparation_started = (
        time.perf_counter()
    )

    analysis_input = (
        prepare_analysis_input(
            raw_input,
            use_supplied_policy_document=(
                args.use_supplied_policy_document
            ),
        )
    )

    preparation_seconds = (
        time.perf_counter()
        - preparation_started
    )

    # -------------------------------------------------------------------------
    # Generate report
    # -------------------------------------------------------------------------

    inference_started = (
        time.perf_counter()
    )

    prediction = generate_report(
        tokenizer,
        model,
        analysis_input,
        max_new_tokens=(
            args.max_new_tokens
        ),
    )

    inference_seconds = (
        time.perf_counter()
        - inference_started
    )

    # Save actual model result.
    Path(
        args.prediction_output
    ).write_text(
        json.dumps(
            prediction,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # Token metrics
    # =========================================================================

    token_correctness = (
        token_sequence_correctness(
            tokenizer,
            expected,
            prediction,
        )
    )

    token_metrics = token_overlap(
        tokenizer,
        expected,
        prediction,
    )

    # =========================================================================
    # Finding metrics
    # =========================================================================

    expected_findings = (
        findings_by_behavior(
            expected
        )
    )

    predicted_findings = (
        findings_by_behavior(
            prediction
        )
    )

    behavior = behavior_metrics(
        expected_findings,
        predicted_findings,
    )

    # =========================================================================
    # Comparison classifications
    # =========================================================================

    comparison_gold = []
    comparison_pred = []

    policy_gold = []
    policy_pred = []

    telemetry_gold = []
    telemetry_pred = []

    confidence_errors = []

    accusation_labels = {
        "observed_only",
        "possible_contradiction",
    }

    predicted_accusations = 0
    false_accusations = 0

    for behavior_name, gold_finding in (
        expected_findings.items()
    ):
        pred_finding = (
            predicted_findings.get(
                behavior_name
            )
        )

        gold_comparison = (
            gold_finding.get(
                "comparison",
                "__missing__",
            )
        )

        if pred_finding is None:
            pred_comparison = (
                "__missing__"
            )

            pred_policy_status = (
                "__missing__"
            )

            pred_telemetry_status = (
                "__missing__"
            )

        else:
            pred_comparison = (
                pred_finding.get(
                    "comparison",
                    "__missing__",
                )
            )

            pred_policy_status = (
                nested_get(
                    pred_finding,
                    "policy",
                    "status",
                    default="__missing__",
                )
            )

            pred_telemetry_status = (
                nested_get(
                    pred_finding,
                    "telemetry",
                    "status",
                    default="__missing__",
                )
            )

            gold_confidence = (
                gold_finding.get(
                    "confidence"
                )
            )

            pred_confidence = (
                pred_finding.get(
                    "confidence"
                )
            )

            if (
                isinstance(
                    gold_confidence,
                    (int, float),
                )
                and isinstance(
                    pred_confidence,
                    (int, float),
                )
            ):
                confidence_errors.append(
                    abs(
                        gold_confidence
                        - pred_confidence
                    )
                )

        comparison_gold.append(
            gold_comparison
        )

        comparison_pred.append(
            pred_comparison
        )

        policy_gold.append(
            nested_get(
                gold_finding,
                "policy",
                "status",
                default="__missing__",
            )
        )

        policy_pred.append(
            pred_policy_status
        )

        telemetry_gold.append(
            nested_get(
                gold_finding,
                "telemetry",
                "status",
                default="__missing__",
            )
        )

        telemetry_pred.append(
            pred_telemetry_status
        )

    # -------------------------------------------------------------------------
    # False accusations
    # -------------------------------------------------------------------------

    for behavior_name, pred_finding in (
        predicted_findings.items()
    ):
        comparison = (
            pred_finding.get(
                "comparison"
            )
        )

        if (
            comparison
            not in accusation_labels
        ):
            continue

        predicted_accusations += 1

        gold_finding = (
            expected_findings.get(
                behavior_name
            )
        )

        if (
            gold_finding is None
            or gold_finding.get(
                "comparison"
            )
            not in accusation_labels
        ):
            false_accusations += 1

    false_accusation_rate = safe_div(
        false_accusations,
        predicted_accusations,
    )

    # =========================================================================
    # Accuracy metrics
    # =========================================================================

    comparison_accuracy = (
        accuracy_score(
            comparison_gold,
            comparison_pred,
        )
        if comparison_gold
        else 0.0
    )

    comparison_macro_f1 = (
        f1_score(
            comparison_gold,
            comparison_pred,
            average="macro",
            zero_division=0,
        )
        if comparison_gold
        else 0.0
    )

    policy_status_accuracy = (
        accuracy_score(
            policy_gold,
            policy_pred,
        )
        if policy_gold
        else 0.0
    )

    telemetry_status_accuracy = (
        accuracy_score(
            telemetry_gold,
            telemetry_pred,
        )
        if telemetry_gold
        else 0.0
    )

    telemetry_evidence_accuracy = (
        evidence_accuracy(
            expected_findings,
            predicted_findings,
        )
    )

    confidence_mae = (
        sum(confidence_errors)
        / len(confidence_errors)
        if confidence_errors
        else 0.0
    )

    confidence_accuracy = max(
        0.0,
        1.0 - confidence_mae,
    )

    exact_json_match = (
        canonical_json(
            expected
        )
        ==
        canonical_json(
            prediction
        )
    )

    # =========================================================================
    # Composite Verity score
    # =========================================================================
    #
    # Semantic correctness matters much more than exact text.
    #
    # 25% comparison accuracy
    # 15% comparison macro-F1
    # 15% finding F1
    # 10% policy-status accuracy
    # 10% telemetry-status accuracy
    # 10% telemetry evidence accuracy
    #  5% token correctness
    #  5% confidence accuracy
    #  5% false-accusation safety
    # =========================================================================

    accusation_safety = (
        1.0
        - false_accusation_rate
    )

    overall_score = (
        comparison_accuracy * 0.25
        + comparison_macro_f1 * 0.15
        + behavior["f1"] * 0.15
        + policy_status_accuracy * 0.10
        + telemetry_status_accuracy * 0.10
        + telemetry_evidence_accuracy * 0.10
        + token_correctness * 0.05
        + confidence_accuracy * 0.05
        + accusation_safety * 0.05
    )

    # =========================================================================
    # Output
    # =========================================================================

    results = {
        "model": {
            "base_model": args.base_model,
            "adapter": args.adapter,
        },

        "score": {
            "overall": overall_score,
            "overall_percent": round(
                overall_score * 100,
                2,
            ),
        },

        "token_metrics": {
            "correctness": token_correctness,
            "precision": token_metrics[
                "precision"
            ],
            "recall": token_metrics[
                "recall"
            ],
            "f1": token_metrics[
                "f1"
            ],
            "matching_tokens": token_metrics[
                "matching_tokens"
            ],
            "expected_tokens": token_metrics[
                "expected_tokens"
            ],
            "predicted_tokens": token_metrics[
                "predicted_tokens"
            ],
        },

        "finding_metrics": behavior,

        "semantic_metrics": {
            "comparison_accuracy":
                comparison_accuracy,

            "comparison_macro_f1":
                comparison_macro_f1,

            "policy_status_accuracy":
                policy_status_accuracy,

            "telemetry_status_accuracy":
                telemetry_status_accuracy,

            "telemetry_evidence_accuracy":
                telemetry_evidence_accuracy,

            "confidence_accuracy":
                confidence_accuracy,

            "confidence_mae":
                confidence_mae,
        },

        "safety": {
            "predicted_accusations":
                predicted_accusations,

            "false_accusations":
                false_accusations,

            "false_accusation_rate":
                false_accusation_rate,

            "accusation_safety":
                accusation_safety,
        },

        "exact_json_match":
            exact_json_match,

        "timing": {
            "model_load_seconds":
                model_load_seconds,

            "input_preparation_seconds":
                preparation_seconds,

            "inference_seconds":
                inference_seconds,
        },
    }

    Path(
        args.output
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # Human-readable benchmark
    # =========================================================================

    print()
    print("=" * 72)
    print("VERITY BENCHMARK RESULTS")
    print("=" * 72)

    print(
        f"Overall Verity Score      : {pct(overall_score)}"
    )

    print()
    print("SEMANTIC ACCURACY")
    print(
        f"Comparison Accuracy       : {pct(comparison_accuracy)}"
    )
    print(
        f"Comparison Macro F1       : {pct(comparison_macro_f1)}"
    )
    print(
        f"Finding Precision         : {pct(behavior['precision'])}"
    )
    print(
        f"Finding Recall            : {pct(behavior['recall'])}"
    )
    print(
        f"Finding F1                : {pct(behavior['f1'])}"
    )
    print(
        f"Policy Status Accuracy    : {pct(policy_status_accuracy)}"
    )
    print(
        f"Telemetry Status Accuracy : {pct(telemetry_status_accuracy)}"
    )
    print(
        f"Telemetry Evidence        : {pct(telemetry_evidence_accuracy)}"
    )

    print()
    print("TOKEN ACCURACY")
    print(
        f"Token Correctness         : {pct(token_correctness)}"
    )
    print(
        f"Token Precision           : {pct(token_metrics['precision'])}"
    )
    print(
        f"Token Recall              : {pct(token_metrics['recall'])}"
    )
    print(
        f"Token F1                  : {pct(token_metrics['f1'])}"
    )

    print()
    print("SAFETY")
    print(
        f"False Accusation Rate     : {pct(false_accusation_rate)}"
    )
    print(
        f"Accusation Safety         : {pct(accusation_safety)}"
    )

    print()
    print("OTHER")
    print(
        f"Confidence Accuracy       : {pct(confidence_accuracy)}"
    )
    print(
        f"Exact JSON Match          : {'YES' if exact_json_match else 'NO'}"
    )
    print(
        f"Expected Findings         : {len(expected_findings)}"
    )
    print(
        f"Predicted Findings        : {len(predicted_findings)}"
    )
    print(
        f"Missing Findings          : {behavior['false_negative']}"
    )
    print(
        f"Extra Findings            : {behavior['false_positive']}"
    )
    print(
        f"Inference Time            : {inference_seconds:.2f}s"
    )

    print("=" * 72)

    # =========================================================================
    # Post-ready output
    # =========================================================================

    print()
    print("POST-READY RESULTS")
    print("-" * 72)

    print(
        f"Verity {args.base_model.split('/')[-1]} benchmark:\n"
        f"\n"
        f"Overall: {pct(overall_score)}\n"
        f"Comparison accuracy: {pct(comparison_accuracy)}\n"
        f"Finding F1: {pct(behavior['f1'])}\n"
        f"Policy accuracy: {pct(policy_status_accuracy)}\n"
        f"Telemetry accuracy: {pct(telemetry_status_accuracy)}\n"
        f"Token correctness: {pct(token_correctness)}\n"
        f"False accusation rate: {pct(false_accusation_rate)}"
    )

    print("-" * 72)

    print()
    print(
        f"Full results written to: {args.output}"
    )

    print(
        f"Prediction written to: {args.prediction_output}"
    )


if __name__ == "__main__":
    main()