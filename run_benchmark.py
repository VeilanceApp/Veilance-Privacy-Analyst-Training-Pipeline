#!/usr/bin/env python3

import argparse
import json
import re
import time
from collections import Counter
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


def nested_get(
    obj,
    *keys,
    default=None,
):
    value = obj

    for key in keys:
        if not isinstance(value, dict):
            return default

        value = value.get(
            key,
            default,
        )

    return value


def normalize_text(value):
    if not isinstance(value, str):
        return ""

    value = value.casefold()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


# =============================================================================
# Behavior normalization
# =============================================================================
#
# Verity is generative. A semantically correct behavior should not become
# one false-positive + one false-negative merely because the generated
# behavior identifier uses a slightly different label.
#
# Keep aliases conservative. Add aliases only when they clearly represent
# the same benchmark concept.
# =============================================================================


BEHAVIOR_ALIASES = {
    # Analytics
    "analytics_tracking":
        "analytics_tracking",

    "analytics_and_measurement":
        "analytics_tracking",

    "performance_and_beacon_measurement":
        "analytics_tracking",

    "google_analytics_activity":
        "analytics_tracking",

    # Browser/device characteristics
    "browser_and_device_characteristics":
        "browser_and_device_characteristics",

    "browser_characteristics":
        "browser_and_device_characteristics",

    "device_characteristics":
        "browser_and_device_characteristics",

    "navigator_characteristics":
        "browser_and_device_characteristics",

    # Cookies
    "cookies":
        "cookies",

    "cookie_access":
        "cookies",

    "cookie_activity":
        "cookies",

    # Storage
    "browser_storage":
        "browser_storage",

    "storage_activity":
        "browser_storage",

    "persistent_storage":
        "browser_storage",

    # Third-party network activity
    "third_party_communications":
        "third_party_communications",

    "third_party_requests":
        "third_party_communications",

    "third_party_network_activity":
        "third_party_communications",

    # Advertising
    "advertising_tracking":
        "advertising_tracking",

    "advertising":
        "advertising_tracking",

    "ad_tracking":
        "advertising_tracking",

    # Tag manager
    "tag_manager_activity":
        "tag_manager_activity",

    "tag_manager":
        "tag_manager_activity",

    "google_tag_manager_activity":
        "tag_manager_activity",

    # Performance API
    "performance_timing":
        "performance_timing",

    "performance_timing_access":
        "performance_timing",

    "browser_performance_timing":
        "performance_timing",

    # Sensors
    "device_sensor_access":
        "device_sensor_access",

    "device_sensors":
        "device_sensor_access",

    "sensor_access":
        "device_sensor_access",
}


def canonical_behavior_name(value):
    if not isinstance(value, str):
        return ""

    normalized = value.strip().casefold()

    return BEHAVIOR_ALIASES.get(
        normalized,
        normalized,
    )


# =============================================================================
# Token metrics
# =============================================================================


def tokenize_json(
    tokenizer,
    value,
):
    return tokenizer.encode(
        canonical_json(value),
        add_special_tokens=False,
    )


def token_overlap(
    tokenizer,
    expected,
    predicted,
):
    """
    Bag-of-token precision / recall / F1.

    Token F1 is the primary token correctness metric.

    This avoids making harmless JSON ordering or insertion differences destroy
    the score for every token that follows.
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

        # This is the headline token correctness number.
        "correctness": f1,

        "matching_tokens": overlap,
        "expected_tokens": len(gold),
        "predicted_tokens": len(pred),
    }


# =============================================================================
# Finding normalization
# =============================================================================


def findings_by_behavior(report):
    findings = report.get(
        "findings",
        []
    )

    if not isinstance(
        findings,
        list,
    ):
        return {}

    result = {}

    for finding in findings:
        if not isinstance(
            finding,
            dict,
        ):
            continue

        behavior = canonical_behavior_name(
            finding.get(
                "behavior"
            )
        )

        if not behavior:
            continue

        # If the model emitted duplicate aliases for the same canonical
        # behavior, keep the first instead of treating them as separate
        # benchmark findings.
        if behavior not in result:
            result[behavior] = finding

    return result


# =============================================================================
# Finding metrics
# =============================================================================


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

    true_positive = len(
        gold & pred
    )

    false_positive = len(
        pred - gold
    )

    false_negative = len(
        gold - pred
    )

    precision = safe_div(
        true_positive,
        true_positive
        + false_positive,
    )

    recall = safe_div(
        true_positive,
        true_positive
        + false_negative,
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
        "true_positive":
            true_positive,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,
    }


# =============================================================================
# Evidence normalization
# =============================================================================


def normalize_evidence_item(value):
    """
    Normalize harmless wording differences.

    We still require the evidence to be substantially the same; this is not
    semantic embedding matching.
    """

    text = normalize_text(
        value
    )

    replacements = {
        "observed for 1 requests":
            "observed for 1 request",

        "observed 1 times":
            "observed 1 time",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return text


def evidence_accuracy(
    expected_findings,
    predicted_findings,
):
    """
    Evidence accuracy over behavior-aligned findings.

    Uses normalized evidence strings and computes micro precision/recall/F1.
    """

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for behavior, gold_finding in (
        expected_findings.items()
    ):
        pred_finding = (
            predicted_findings.get(
                behavior
            )
        )

        gold_evidence = nested_get(
            gold_finding,
            "telemetry",
            "evidence",
            default=[],
        )

        if not isinstance(
            gold_evidence,
            list,
        ):
            gold_evidence = []

        gold_set = {
            normalize_evidence_item(
                item
            )
            for item in gold_evidence
            if normalize_evidence_item(
                item
            )
        }

        if pred_finding is None:
            true_positive += 0
            false_negative += len(
                gold_set
            )
            continue

        pred_evidence = nested_get(
            pred_finding,
            "telemetry",
            "evidence",
            default=[],
        )

        if not isinstance(
            pred_evidence,
            list,
        ):
            pred_evidence = []

        pred_set = {
            normalize_evidence_item(
                item
            )
            for item in pred_evidence
            if normalize_evidence_item(
                item
            )
        }

        true_positive += len(
            gold_set
            & pred_set
        )

        false_positive += len(
            pred_set
            - gold_set
        )

        false_negative += len(
            gold_set
            - pred_set
        )

    precision = safe_div(
        true_positive,
        true_positive
        + false_positive,
    )

    recall = safe_div(
        true_positive,
        true_positive
        + false_negative,
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
        "true_positive":
            true_positive,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,
    }


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

    load_started = (
        time.perf_counter()
    )

    tokenizer, model = load_model(
        args.base_model,
        args.adapter,
    )

    model_load_seconds = (
        time.perf_counter()
        - load_started
    )

    # -------------------------------------------------------------------------
    # Prepare inference input
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
    # Run inference
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

    # -------------------------------------------------------------------------
    # Save model prediction
    # -------------------------------------------------------------------------

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

    token_metrics = token_overlap(
        tokenizer,
        expected,
        prediction,
    )

    token_correctness = (
        token_metrics["f1"]
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
    # Semantic metrics
    # =========================================================================
    #
    # IMPORTANT:
    #
    # Semantic classification metrics are calculated ONLY for behavior findings
    # that exist in both reports.
    #
    # Missing/extra findings are already penalized by finding precision/recall/F1.
    # Penalizing them again as "__missing__" comparison/policy/telemetry labels
    # double-counts the same error.
    # =========================================================================

    shared_behaviors = sorted(
        set(
            expected_findings
        )
        & set(
            predicted_findings
        )
    )

    comparison_gold = []
    comparison_pred = []

    policy_gold = []
    policy_pred = []

    telemetry_gold = []
    telemetry_pred = []

    confidence_errors = []

    for behavior_name in shared_behaviors:
        gold_finding = (
            expected_findings[
                behavior_name
            ]
        )

        pred_finding = (
            predicted_findings[
                behavior_name
            ]
        )

        comparison_gold.append(
            gold_finding.get(
                "comparison",
                "__missing__",
            )
        )

        comparison_pred.append(
            pred_finding.get(
                "comparison",
                "__missing__",
            )
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
            nested_get(
                pred_finding,
                "policy",
                "status",
                default="__missing__",
            )
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
            and not isinstance(
                gold_confidence,
                bool,
            )
            and isinstance(
                pred_confidence,
                (int, float),
            )
            and not isinstance(
                pred_confidence,
                bool,
            )
        ):
            confidence_errors.append(
                abs(
                    float(
                        gold_confidence
                    )
                    - float(
                        pred_confidence
                    )
                )
            )

    # =========================================================================
    # Classification scores
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

    # =========================================================================
    # Evidence metrics
    # =========================================================================

    evidence = evidence_accuracy(
        expected_findings,
        predicted_findings,
    )

    # =========================================================================
    # Confidence
    # =========================================================================

    confidence_mae = (
        sum(
            confidence_errors
        )
        / len(
            confidence_errors
        )
        if confidence_errors
        else 0.0
    )

    confidence_accuracy = max(
        0.0,
        1.0
        - confidence_mae,
    )

    # =========================================================================
    # False accusations
    # =========================================================================

    accusation_labels = {
        "observed_only",
        "possible_contradiction",
    }

    predicted_accusations = 0
    false_accusations = 0

    for (
        behavior_name,
        pred_finding,
    ) in predicted_findings.items():

        pred_comparison = (
            pred_finding.get(
                "comparison"
            )
        )

        if (
            pred_comparison
            not in accusation_labels
        ):
            continue

        predicted_accusations += 1

        gold_finding = (
            expected_findings.get(
                behavior_name
            )
        )

        if gold_finding is None:
            false_accusations += 1
            continue

        if (
            gold_finding.get(
                "comparison"
            )
            not in accusation_labels
        ):
            false_accusations += 1

    false_accusation_rate = (
        safe_div(
            false_accusations,
            predicted_accusations,
        )
    )

    accusation_safety = (
        1.0
        - false_accusation_rate
    )

    # =========================================================================
    # Exact JSON
    # =========================================================================

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
    # Each major capability is represented approximately once.
    #
    # 30% Finding F1
    #     Did Verity identify the correct behaviors?
    #
    # 25% Comparison accuracy
    #     Did Verity classify policy-vs-telemetry correctly?
    #
    # 15% Policy status accuracy
    #     Did Verity correctly interpret disclosure status?
    #
    # 10% Telemetry evidence F1
    #     Did Verity ground its findings in the right evidence?
    #
    # 10% Token F1
    #     How similar is the complete structured answer?
    #
    # 10% Safety
    #     Does Verity avoid unsupported disclosure-gap/contradiction claims?
    #
    # We intentionally do NOT include:
    #
    # - comparison macro F1 in the composite
    # - telemetry-status accuracy in the composite
    # - confidence accuracy in the composite
    #
    # They remain diagnostic metrics. Including all of them would repeatedly
    # punish the same underlying mistake.
    # =========================================================================

    overall_score = (
        behavior["f1"]
        * 0.30

        + comparison_accuracy
        * 0.25

        + policy_status_accuracy
        * 0.15

        + evidence["f1"]
        * 0.10

        + token_correctness
        * 0.10

        + accusation_safety
        * 0.10
    )

    # =========================================================================
    # Results JSON
    # =========================================================================

    results = {
        "model": {
            "base_model":
                args.base_model,

            "adapter":
                args.adapter,
        },

        "score": {
            "overall":
                overall_score,

            "overall_percent":
                round(
                    overall_score
                    * 100,
                    2,
                ),
        },

        "token_metrics": {
            "correctness":
                token_correctness,

            "precision":
                token_metrics[
                    "precision"
                ],

            "recall":
                token_metrics[
                    "recall"
                ],

            "f1":
                token_metrics[
                    "f1"
                ],

            "matching_tokens":
                token_metrics[
                    "matching_tokens"
                ],

            "expected_tokens":
                token_metrics[
                    "expected_tokens"
                ],

            "predicted_tokens":
                token_metrics[
                    "predicted_tokens"
                ],
        },

        "finding_metrics":
            behavior,

        "semantic_metrics": {
            "shared_findings":
                len(
                    shared_behaviors
                ),

            "comparison_accuracy":
                comparison_accuracy,

            "comparison_macro_f1":
                comparison_macro_f1,

            "policy_status_accuracy":
                policy_status_accuracy,

            "telemetry_status_accuracy":
                telemetry_status_accuracy,

            "telemetry_evidence_precision":
                evidence[
                    "precision"
                ],

            "telemetry_evidence_recall":
                evidence[
                    "recall"
                ],

            "telemetry_evidence_f1":
                evidence[
                    "f1"
                ],

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

        "finding_counts": {
            "expected":
                len(
                    expected_findings
                ),

            "predicted":
                len(
                    predicted_findings
                ),

            "shared":
                len(
                    shared_behaviors
                ),

            "missing":
                behavior[
                    "false_negative"
                ],

            "extra":
                behavior[
                    "false_positive"
                ],
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

    # -------------------------------------------------------------------------
    # Save benchmark
    # -------------------------------------------------------------------------

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
    # Human-readable output
    # =========================================================================

    print()
    print("=" * 72)
    print("VERITY BENCHMARK RESULTS")
    print("=" * 72)

    print(
        f"Overall Verity Score      : {pct(overall_score)}"
    )

    print()
    print("FINDING QUALITY")

    print(
        f"Finding Precision         : {pct(behavior['precision'])}"
    )

    print(
        f"Finding Recall            : {pct(behavior['recall'])}"
    )

    print(
        f"Finding F1                : {pct(behavior['f1'])}"
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
        f"Policy Status Accuracy    : {pct(policy_status_accuracy)}"
    )

    print(
        f"Telemetry Status Accuracy : {pct(telemetry_status_accuracy)}"
    )

    print(
        f"Evidence Precision        : {pct(evidence['precision'])}"
    )

    print(
        f"Evidence Recall           : {pct(evidence['recall'])}"
    )

    print(
        f"Evidence F1               : {pct(evidence['f1'])}"
    )

    print()
    print("TOKEN ACCURACY")

    print(
        f"Token Correctness (F1)    : {pct(token_correctness)}"
    )

    print(
        f"Token Precision           : {pct(token_metrics['precision'])}"
    )

    print(
        f"Token Recall              : {pct(token_metrics['recall'])}"
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
    print("DIAGNOSTICS")

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
        f"Shared Findings           : {len(shared_behaviors)}"
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

    model_name = (
        args.base_model
        .split("/")[-1]
    )

    print()
    print("POST-READY RESULTS")
    print("-" * 72)

    print(
        f"Verity {model_name} benchmark:\n"
        f"\n"
        f"Overall task score: {pct(overall_score)}\n"
        f"Finding F1: {pct(behavior['f1'])}\n"
        f"Comparison accuracy: {pct(comparison_accuracy)}\n"
        f"Policy accuracy: {pct(policy_status_accuracy)}\n"
        f"Evidence F1: {pct(evidence['f1'])}\n"
        f"Token F1: {pct(token_correctness)}\n"
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