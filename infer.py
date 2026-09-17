import argparse
import json
import logging
import re
import sys
import time

from pathlib import Path
from typing import Optional

from input_normalization import (
    build_analysis_input,
    normalize_runtime_input,
    validate_policy_document,
)
from policy_retrieval import PolicyRetriever
from prompt import SYSTEM_PROMPT, build_user_prompt
from schema import recompute_counts, validate_report


LOGGER_NAME = "verity.inference"
logger = logging.getLogger(LOGGER_NAME)


# =============================================================================
# Logging
# =============================================================================


def _normalize_log_level(level) -> int:
    if isinstance(level, int):
        return level

    if not isinstance(level, str):
        raise ValueError(
            "log level must be a logging level name or integer"
        )

    normalized = level.strip().upper()
    value = getattr(logging, normalized, None)

    if not isinstance(value, int):
        raise ValueError(
            f"invalid log level {level!r}; "
            "use DEBUG, INFO, WARNING, ERROR, or CRITICAL"
        )

    return value


def configure_logging(
    *,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configure Verity inference logging.

    Logs always go to stderr so stdout remains clean for JSON output.

    If log_file is supplied, the same logs are also written to that file.
    """

    level = _normalize_log_level(log_level)

    if log_format is None:
        log_format = (
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        )

    formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("verity")
    root.setLevel(level)
    root.propagate = False

    # Avoid duplicate handlers if configure_logging() is called repeatedly.
    root.handlers.clear()

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if log_file:
        log_path = Path(log_file)

        if log_path.parent != Path("."):
            log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    configured_logger = logging.getLogger(LOGGER_NAME)
    configured_logger.setLevel(level)

    configured_logger.info(
        "Logging initialized level=%s file=%s",
        logging.getLevelName(level),
        log_file or "<none>",
    )

    return configured_logger


def _format_bytes(value: int) -> str:
    value = int(value)

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    )

    size = float(value)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{value} B"


def _json_size(value) -> int:
    try:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    except Exception:
        return 0


def _cuda_memory_summary(torch) -> dict:
    """Return lightweight CUDA memory information for logging."""

    if not torch.cuda.is_available():
        return {}

    try:
        device = torch.cuda.current_device()

        return {
            "device": device,
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device),
        }

    except Exception:
        return {}


def _log_cuda_memory(torch, prefix: str, *, level=logging.DEBUG) -> None:
    memory = _cuda_memory_summary(torch)

    if not memory:
        return

    logger.log(
        level,
        "%s CUDA memory "
        "device=%d allocated=%s reserved=%s "
        "max_allocated=%s max_reserved=%s",
        prefix,
        memory["device"],
        _format_bytes(memory["allocated"]),
        _format_bytes(memory["reserved"]),
        _format_bytes(memory["max_allocated"]),
        _format_bytes(memory["max_reserved"]),
    )


# =============================================================================
# Model loading
# =============================================================================


def load_model(base: str, adapter: str):
    started = time.perf_counter()

    logger.info(
        "Starting model load base_model=%s adapter=%s",
        base,
        adapter,
    )

    try:
        import torch
        import transformers
        import peft

        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        logger.debug(
            "Library versions torch=%s transformers=%s peft=%s",
            getattr(torch, "__version__", "unknown"),
            getattr(transformers, "__version__", "unknown"),
            getattr(peft, "__version__", "unknown"),
        )

        if not torch.cuda.is_available():
            logger.error(
                "CUDA is unavailable; configured inference path requires a GPU"
            )

            raise RuntimeError(
                "CUDA GPU required for the configured 4-bit inference path"
            )

        device_count = torch.cuda.device_count()

        logger.info(
            "CUDA available device_count=%d",
            device_count,
        )

        for index in range(device_count):
            try:
                properties = torch.cuda.get_device_properties(index)

                logger.info(
                    "CUDA device index=%d name=%r "
                    "total_memory=%s capability=%d.%d",
                    index,
                    properties.name,
                    _format_bytes(properties.total_memory),
                    properties.major,
                    properties.minor,
                )

            except Exception as exc:
                logger.debug(
                    "Unable to inspect CUDA device index=%d error=%s",
                    index,
                    exc,
                )

        bf16_supported = torch.cuda.is_bf16_supported()

        dtype = (
            torch.bfloat16
            if bf16_supported
            else torch.float16
        )

        logger.info(
            "Inference dtype selected dtype=%s bf16_supported=%s",
            dtype,
            bf16_supported,
        )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

        logger.info(
            "4-bit quantization configured "
            "type=nf4 double_quant=true compute_dtype=%s",
            dtype,
        )

        _log_cuda_memory(
            torch,
            "Before tokenizer/model load",
        )

        tokenizer_started = time.perf_counter()

        logger.info(
            "Loading tokenizer adapter=%s",
            adapter,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            adapter,
            use_fast=True,
        )

        logger.info(
            "Tokenizer loaded duration=%.3fs "
            "class=%s vocab_size=%s "
            "eos_token_id=%s pad_token_id=%s",
            time.perf_counter() - tokenizer_started,
            tokenizer.__class__.__name__,
            getattr(tokenizer, "vocab_size", "unknown"),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )

        base_started = time.perf_counter()

        logger.info(
            "Loading quantized base model base=%s",
            base,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quantization,
            dtype=dtype,
            device_map="auto",
        )

        logger.info(
            "Base model loaded duration=%.3fs "
            "class=%s",
            time.perf_counter() - base_started,
            base_model.__class__.__name__,
        )

        _log_cuda_memory(
            torch,
            "After base model load",
            level=logging.INFO,
        )

        adapter_started = time.perf_counter()

        logger.info(
            "Loading PEFT adapter adapter=%s",
            adapter,
        )

        model = PeftModel.from_pretrained(
            base_model,
            adapter,
        )

        logger.info(
            "PEFT adapter loaded duration=%.3fs "
            "class=%s",
            time.perf_counter() - adapter_started,
            model.__class__.__name__,
        )

        model.eval()

        logger.info(
            "Model placed in evaluation mode"
        )

        try:
            logger.info(
                "Model primary device=%s",
                model.device,
            )
        except Exception:
            logger.debug(
                "Model does not expose a single primary device"
            )

        try:
            device_map = getattr(
                base_model,
                "hf_device_map",
                None,
            )

            if device_map:
                logger.debug(
                    "Model HF device map=%s",
                    device_map,
                )

        except Exception as exc:
            logger.debug(
                "Unable to inspect model device map error=%s",
                exc,
            )

        _log_cuda_memory(
            torch,
            "After adapter load",
            level=logging.INFO,
        )

        logger.info(
            "Model load complete total_duration=%.3fs",
            time.perf_counter() - started,
        )

        return tokenizer, model

    except Exception:
        logger.exception(
            "Model loading failed "
            "base_model=%s adapter=%s duration=%.3fs",
            base,
            adapter,
            time.perf_counter() - started,
        )

        raise


# =============================================================================
# JSON parsing
# =============================================================================


def parse_json(text: str) -> dict:
    started = time.perf_counter()

    if not isinstance(text, str):
        logger.error(
            "Model output is not text type=%s",
            type(text).__name__,
        )

        raise ValueError(
            "model output must be text"
        )

    original_length = len(text)

    logger.info(
        "Parsing generated model output chars=%d",
        original_length,
    )

    text = text.strip()

    think_match = re.match(
        r"^<think>.*?</think>\s*",
        text,
        flags=re.S,
    )

    if think_match:
        removed_chars = len(
            think_match.group(0)
        )

        logger.warning(
            "Model emitted a <think> block despite "
            "non-thinking inference; removing chars=%d",
            removed_chars,
        )

        text = text[
            think_match.end():
        ]

    text = text.strip()

    if text.startswith("```") or text.endswith("```"):
        logger.error(
            "Model returned Markdown code fencing "
            "instead of raw JSON chars=%d",
            len(text),
        )

        logger.debug(
            "Malformed model-output preview=%r",
            text[:2_000],
        )

        raise ValueError(
            "model returned a Markdown code fence instead of raw JSON"
        )

    try:
        value = json.loads(text)

    except json.JSONDecodeError as exc:
        logger.error(
            "Model output JSON decoding failed "
            "line=%d column=%d position=%d error=%s",
            exc.lineno,
            exc.colno,
            exc.pos,
            exc.msg,
        )

        logger.debug(f"=== START RAW TEXT ===\n{text}\n=== END RAW TEXT ===")

        start = max(
            0,
            exc.pos - 500,
        )

        end = min(
            len(text),
            exc.pos + 500,
        )

        logger.debug(
            "JSON parse failure surrounding output=%r",
            text[start:end],
        )

        raise

    if not isinstance(value, dict):
        logger.error(
            "Parsed model JSON has invalid root type=%s",
            type(value).__name__,
        )

        raise ValueError(
            "model output must be one JSON object"
        )

    logger.info(
        "Model output parsed successfully "
        "keys=%s duration=%.3fs",
        sorted(value.keys()),
        time.perf_counter() - started,
    )

    return value


# =============================================================================
# Helpers
# =============================================================================


def _append_unique(
    values: list,
    additions: list[str],
) -> list:
    output = [
        item
        for item in values
        if isinstance(item, str)
    ]

    before = len(output)

    for addition in additions:
        if (
            addition
            and addition not in output
        ):
            output.append(addition)

    logger.debug(
        "Appended unique limitations "
        "initial=%d additions=%d final=%d",
        before,
        len(additions),
        len(output),
    )

    return output


def _section_key(value: str) -> str:
    """Normalize harmless punctuation/spacing differences in policy headings."""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.casefold(),
    ).strip()


def _downgrade_finding_for_policy_failure(
    finding: dict,
    *,
    explanation: str,
) -> None:
    """Make a finding schema-valid without inventing missing policy evidence."""

    finding["policy"] = {
        "status": "unknown",
        "evidence": "",
        "section": "",
    }

    finding["comparison"] = (
        "indeterminate"
    )

    finding["severity"] = (
        "informational"
    )

    confidence = finding.get(
        "confidence"
    )

    if (
        isinstance(
            confidence,
            (int, float),
        )
        and not isinstance(
            confidence,
            bool,
        )
    ):
        finding["confidence"] = min(
            float(confidence),
            0.5,
        )

    finding["explanation"] = explanation


# =============================================================================
# Policy grounding
# =============================================================================


def _repair_policy_grounding(
    report: dict,
    policy_document: dict,
) -> int:
    """Conservatively repair missing or ungrounded model policy evidence.

    Returns the number of findings downgraded because the model omitted the
    policy object or its disclosure citation could not be grounded in a
    Playwright-extracted section.
    """

    started = time.perf_counter()

    headings = [
        section["heading"].strip()
        for section
        in policy_document.get(
            "sections",
            [],
        )
        if isinstance(section, dict)
        and isinstance(
            section.get("heading"),
            str,
        )
        and section["heading"].strip()
    ]

    by_key = {
        _section_key(heading): heading
        for heading in headings
    }

    disclosure_statuses = {
        "explicitly_disclosed",
        "broadly_disclosed",
        "implicitly_disclosed",
        "contradicted",
    }

    findings = report.get(
        "findings",
        [],
    )

    logger.info(
        "Starting policy-grounding repair "
        "findings=%d extracted_headings=%d",
        len(findings)
        if isinstance(findings, list)
        else 0,
        len(headings),
    )

    logger.debug(
        "Available policy headings=%s",
        headings,
    )

    downgraded = 0
    missing_policy_objects = 0
    repaired = 0
    unsupported_optional_citations = 0

    for finding_index, finding in enumerate(
        findings
    ):
        if not isinstance(finding, dict):
            logger.debug(
                "Skipping grounding check finding=%d "
                "reason=invalid_finding_type",
                finding_index,
            )
            continue

        if not isinstance(
            finding.get("policy"),
            dict,
        ):
            old_comparison = finding.get(
                "comparison"
            )

            old_confidence = finding.get(
                "confidence"
            )

            _downgrade_finding_for_policy_failure(
                finding,
                explanation=(
                    "The telemetry evidence remains available, but the policy "
                    "comparison is indeterminate because the model omitted the "
                    "required policy evidence object from this finding."
                ),
            )

            downgraded += 1
            missing_policy_objects += 1

            logger.warning(
                "Repaired missing policy object "
                "finding=%d old_comparison=%s "
                "old_confidence=%s new_confidence=%s "
                "comparison=indeterminate",
                finding_index,
                old_comparison,
                old_confidence,
                finding.get("confidence"),
            )
            continue

        policy = finding["policy"]
        status = policy.get("status")
        evidence = policy.get("evidence")
        section_value = policy.get("section")

        resolved = []
        unresolved = False

        logger.debug(
            "Grounding finding=%d "
            "comparison=%s policy_status=%s "
            "section=%r evidence_chars=%d",
            finding_index,
            finding.get("comparison"),
            status,
            section_value,
            (
                len(evidence)
                if isinstance(evidence, str)
                else 0
            ),
        )

        if (
            isinstance(section_value, str)
            and section_value.strip()
        ):
            cited_values = [
                item.strip()
                for item
                in section_value.split(";")
                if item.strip()
            ]

            logger.debug(
                "Finding=%d cited_policy_sections=%s",
                finding_index,
                cited_values,
            )

            for cited in cited_values:
                key = _section_key(
                    cited
                )

                canonical = by_key.get(
                    key
                )

                if canonical is not None:
                    logger.debug(
                        "Finding=%d exact heading match "
                        "generated=%r canonical=%r",
                        finding_index,
                        cited,
                        canonical,
                    )

                if (
                    canonical is None
                    and len(key) >= 8
                ):
                    candidates = [
                        heading
                        for heading
                        in headings
                        if (
                            key
                            in _section_key(
                                heading
                            )
                            or _section_key(
                                heading
                            )
                            in key
                        )
                    ]

                    logger.debug(
                        "Finding=%d fuzzy heading resolution "
                        "generated=%r candidates=%s",
                        finding_index,
                        cited,
                        candidates,
                    )

                    if len(candidates) == 1:
                        canonical = (
                            candidates[0]
                        )

                if canonical is None:
                    logger.warning(
                        "Unable to ground generated policy "
                        "section finding=%d status=%s "
                        "generated_section=%r",
                        finding_index,
                        status,
                        cited,
                    )

                    unresolved = True
                    break

                if canonical not in resolved:
                    resolved.append(
                        canonical
                    )

        elif status in disclosure_statuses:
            logger.warning(
                "Disclosure-dependent finding has no "
                "policy section finding=%d status=%s",
                finding_index,
                status,
            )

            unresolved = True

        if (
            status in disclosure_statuses
            and (
                not isinstance(
                    evidence,
                    str,
                )
                or not evidence.strip()
            )
        ):
            logger.warning(
                "Disclosure-dependent finding has no "
                "policy evidence finding=%d status=%s",
                finding_index,
                status,
            )

            unresolved = True

        if not unresolved:
            if resolved:
                canonical_value = (
                    "; ".join(resolved)
                )

                if (
                    canonical_value
                    != section_value
                ):
                    logger.info(
                        "Repaired policy heading citation "
                        "finding=%d before=%r after=%r",
                        finding_index,
                        section_value,
                        canonical_value,
                    )

                    repaired += 1

                policy["section"] = (
                    canonical_value
                )

            continue

        if status not in disclosure_statuses:
            logger.info(
                "Clearing unsupported optional policy "
                "citation finding=%d status=%s",
                finding_index,
                status,
            )

            policy["evidence"] = ""
            policy["section"] = ""

            unsupported_optional_citations += 1
            continue

        old_confidence = finding.get(
            "confidence"
        )

        _downgrade_finding_for_policy_failure(
            finding,
            explanation=(
                "The telemetry evidence remains available, but the policy comparison "
                "is indeterminate because the generated policy-section citation could "
                "not be grounded in a section extracted by Playwright."
            ),
        )

        downgraded += 1

        logger.warning(
            "Downgraded ungrounded finding "
            "finding=%d old_status=%s "
            "old_confidence=%s new_confidence=%s "
            "comparison=indeterminate",
            finding_index,
            status,
            old_confidence,
            finding.get("confidence"),
        )

    logger.info(
        "Policy-grounding repair complete "
        "findings=%d repaired_headings=%d "
        "missing_policy_objects=%d "
        "optional_citations_cleared=%d "
        "downgraded=%d duration=%.3fs",
        len(findings)
        if isinstance(findings, list)
        else 0,
        repaired,
        missing_policy_objects,
        unsupported_optional_citations,
        downgraded,
        time.perf_counter() - started,
    )

    return downgraded


def _validate_policy_grounding(
    report: dict,
    policy_document: dict,
) -> None:
    """Reject section citations that are not present in retrieved policy text."""

    started = time.perf_counter()

    if not (
        policy_document["found"]
        and policy_document["applicable"]
    ):
        logger.info(
            "Skipping policy-grounding validation "
            "because no applicable policy is available"
        )
        return

    headings = {
        section["heading"]
        .strip()
        .casefold()
        for section
        in policy_document["sections"]
        if section["heading"].strip()
    }

    disclosure_statuses = {
        "explicitly_disclosed",
        "broadly_disclosed",
        "implicitly_disclosed",
        "contradicted",
    }

    findings = report.get(
        "findings",
        [],
    )

    logger.info(
        "Validating policy grounding "
        "findings=%d valid_headings=%d",
        len(findings),
        len(headings),
    )

    for index, finding in enumerate(
        findings
    ):
        policy = (
            finding.get("policy")
            if isinstance(
                finding,
                dict,
            )
            else None
        )

        if not isinstance(policy, dict):
            logger.debug(
                "Skipping policy grounding validation "
                "finding=%d reason=no_policy_object",
                index,
            )
            continue

        status = policy.get(
            "status"
        )

        evidence = policy.get(
            "evidence"
        )

        section = policy.get(
            "section"
        )

        logger.debug(
            "Validating finding=%d "
            "status=%s section=%r evidence_chars=%d",
            index,
            status,
            section,
            (
                len(evidence)
                if isinstance(
                    evidence,
                    str,
                )
                else 0
            ),
        )

        if (
            status in disclosure_statuses
            and (
                not isinstance(
                    evidence,
                    str,
                )
                or not evidence.strip()
            )
        ):
            logger.error(
                "Policy grounding validation failed "
                "finding=%d reason=missing_evidence "
                "status=%s",
                index,
                status,
            )

            raise ValueError(
                f"findings/{index}/policy/evidence "
                f"is required for {status}"
            )

        if (
            isinstance(section, str)
            and section.strip()
        ):
            cited = [
                item.strip().casefold()
                for item
                in section.split(";")
                if item.strip()
            ]

            unknown = [
                item
                for item in cited
                if item not in headings
            ]

            if not cited or unknown:
                logger.error(
                    "Policy grounding validation failed "
                    "finding=%d reason=unknown_section "
                    "cited=%s unknown=%s",
                    index,
                    cited,
                    unknown,
                )

                raise ValueError(
                    f"findings/{index}/policy/section "
                    "cites a section that was not extracted by Playwright"
                )

        elif status in disclosure_statuses:
            logger.error(
                "Policy grounding validation failed "
                "finding=%d reason=missing_section "
                "status=%s",
                index,
                status,
            )

            raise ValueError(
                f"findings/{index}/policy/section "
                f"is required for {status}"
            )

    logger.info(
        "Policy grounding validation passed "
        "findings=%d duration=%.3fs",
        len(findings),
        time.perf_counter() - started,
    )


# =============================================================================
# Report finalization
# =============================================================================


def finalize_report(
    report: dict,
    analysis_input: dict,
) -> dict:
    """Apply authoritative host fields and deterministic derived values."""

    started = time.perf_counter()

    logger.info(
        "Starting report finalization"
    )

    policy_document = (
        analysis_input[
            "policy_document"
        ]
    )

    visit = analysis_input[
        "visit"
    ]

    report["domain"] = (
        analysis_input[
            "domain_url"
        ]
    )

    policy_available = bool(
        policy_document["found"]
        and policy_document["applicable"]
    )

    logger.info(
        "Authoritative host context "
        "domain=%s policy_available=%s "
        "policy_url=%s "
        "visit_observed_at=%s "
        "visit_duration_seconds=%s",
        analysis_input[
            "domain_url"
        ],
        policy_available,
        (
            policy_document.get("url")
            or "<none>"
        ),
        visit.get("observed_at"),
        visit.get(
            "duration_seconds"
        ),
    )

    report["privacy_policy"] = {
        "url": (
            policy_document["url"]
            if policy_available
            else ""
        ),
        "found": policy_available,
        "applicable": policy_available,
    }

    report["visit"] = {
        "observed_at":
            visit.get(
                "observed_at"
            ),
        "duration_seconds":
            visit[
                "duration_seconds"
            ],
    }

    findings = report.get(
        "findings"
    )

    if not isinstance(
        findings,
        list,
    ):
        logger.error(
            "Model output findings has invalid type=%s",
            type(findings).__name__,
        )

        raise ValueError(
            "model output findings must be an array"
        )

    logger.info(
        "Model produced findings=%d",
        len(findings),
    )

    if not policy_available:
        logger.warning(
            "No applicable privacy policy available; "
            "forcing all comparable findings to indeterminate"
        )

        forced_indeterminate = 0

        for finding_index, finding in enumerate(
            findings
        ):
            if not isinstance(
                finding,
                dict,
            ):
                logger.debug(
                    "Skipping no-policy rewrite finding=%d "
                    "reason=invalid_finding_type",
                    finding_index,
                )
                continue

            old_comparison = finding.get(
                "comparison"
            )

            old_confidence = finding.get(
                "confidence"
            )

            _downgrade_finding_for_policy_failure(
                finding,
                explanation=(
                    "No applicable privacy policy was available, so this behavior "
                    "cannot be reliably compared with policy disclosure."
                ),
            )

            forced_indeterminate += 1

            logger.debug(
                "Forced finding indeterminate "
                "finding=%d old_comparison=%s "
                "old_confidence=%s new_confidence=%s",
                finding_index,
                old_comparison,
                old_confidence,
                finding.get(
                    "confidence"
                ),
            )

        logger.info(
            "No-policy rewrite complete "
            "rewritten_findings=%d",
            forced_indeterminate,
        )

    grounding_downgrades = (
        _repair_policy_grounding(
            report,
            policy_document,
        )
        if policy_available
        else 0
    )

    analysis = report.get(
        "analysis"
    )

    if not isinstance(
        analysis,
        dict,
    ):
        logger.error(
            "Model output analysis has invalid type=%s",
            type(analysis).__name__,
        )

        raise ValueError(
            "model output analysis must be an object"
        )

    if not policy_available:
        observed_findings = sum(
            isinstance(
                finding,
                dict,
            )
            and isinstance(
                finding.get(
                    "telemetry"
                ),
                dict,
            )
            and finding[
                "telemetry"
            ].get(
                "status"
            )
            == "observed"
            for finding in findings
        )

        analysis["summary"] = (
            f"Veilance identified {observed_findings} grouped observed behaviors "
            "during this visit, but no applicable privacy policy was available, "
            "so the comparisons are indeterminate."
        )

        logger.info(
            "Rebuilt analysis summary for missing policy "
            "observed_findings=%d",
            observed_findings,
        )

    elif grounding_downgrades:
        noun = (
            "finding was"
            if grounding_downgrades == 1
            else "findings were"
        )

        analysis["summary"] = (
            f"During this visit, Veilance produced {len(findings)} grouped findings. "
            f"{grounding_downgrades} {noun} marked indeterminate because the "
            "generated policy evidence was missing or could not be grounded in the "
            "Playwright-extracted policy sections. The remaining findings passed "
            "policy-grounding validation."
        )

        logger.warning(
            "Rebuilt analysis summary because of "
            "grounding downgrades count=%d",
            grounding_downgrades,
        )

    counts = recompute_counts(
        findings
    )

    analysis["counts"] = counts

    logger.info(
        "Recomputed comparison counts=%s",
        counts,
    )

    confidences = [
        finding.get(
            "confidence"
        )
        for finding in findings
        if isinstance(
            finding,
            dict,
        )
        and isinstance(
            finding.get(
                "confidence"
            ),
            (int, float),
        )
        and not isinstance(
            finding.get(
                "confidence"
            ),
            bool,
        )
    ]

    analysis[
        "overall_confidence"
    ] = (
        round(
            sum(confidences)
            / len(confidences),
            2,
        )
        if confidences
        else 0.0
    )

    logger.info(
        "Recomputed overall confidence "
        "value=%.2f contributing_findings=%d",
        analysis[
            "overall_confidence"
        ],
        len(confidences),
    )

    limitations = report.get(
        "important_limitations"
    )

    if not isinstance(
        limitations,
        list,
    ):
        logger.error(
            "Model output important_limitations "
            "has invalid type=%s",
            type(limitations).__name__,
        )

        raise ValueError(
            "model output important_limitations must be an array"
        )

    host_limitations = (
        policy_document.get(
            "limitations",
            [],
        )
    )

    if grounding_downgrades:
        host_limitations = (
            host_limitations
            + [
                "One or more findings contained missing or ungrounded generated "
                "policy evidence; affected findings were marked indeterminate."
            ]
        )

    report[
        "important_limitations"
    ] = _append_unique(
        limitations,
        host_limitations,
    )

    logger.info(
        "Final report limitations "
        "model_limitations=%d "
        "host_limitations=%d final=%d",
        len(limitations),
        len(host_limitations),
        len(
            report[
                "important_limitations"
            ]
        ),
    )

    _validate_policy_grounding(
        report,
        policy_document,
    )

    validation_started = (
        time.perf_counter()
    )

    logger.info(
        "Running strict report schema validation"
    )

    try:
        validated = validate_report(
            report
        )

    except Exception:
        logger.exception(
            "Strict report schema validation failed"
        )
        raise

    logger.info(
        "Strict report schema validation passed "
        "duration=%.3fs",
        time.perf_counter()
        - validation_started,
    )

    logger.info(
        "Report finalization complete "
        "findings=%d grounding_downgrades=%d "
        "overall_confidence=%.2f duration=%.3fs",
        len(findings),
        grounding_downgrades,
        analysis[
            "overall_confidence"
        ],
        time.perf_counter()
        - started,
    )

    return validated


# =============================================================================
# Engine
# =============================================================================


class PrivacyPolicyComparisonEngine:
    """Reusable local equivalent of the existing ChatGPT connector call."""

    def __init__(
        self,
        *,
        adapter: str,
        base_model: str = "Qwen/Qwen3-1.7B",
        retriever: Optional[
            PolicyRetriever
        ] = None,
        max_new_tokens: int = 3_200,
    ):
        started = time.perf_counter()

        logger.info(
            "Initializing PrivacyPolicyComparisonEngine "
            "base_model=%s adapter=%s max_new_tokens=%d",
            base_model,
            adapter,
            max_new_tokens,
        )

        self.tokenizer, self.model = load_model(
            base_model,
            adapter,
        )

        if retriever is None:
            logger.info(
                "No PolicyRetriever supplied; "
                "creating default retriever"
            )

            retriever = PolicyRetriever(
                logger=logging.getLogger(
                    "verity.policy_retrieval"
                ),
                log_level=logging.getLevelName(
                    logger.getEffectiveLevel()
                ),
            )

        else:
            logger.info(
                "Using caller-provided PolicyRetriever"
            )

        self.retriever = retriever
        self.max_new_tokens = max_new_tokens

        logger.info(
            "PrivacyPolicyComparisonEngine initialized "
            "duration=%.3fs",
            time.perf_counter()
            - started,
        )

    def privacy_policy_comparison(
        self,
        payload: dict,
    ) -> dict:
        """Accept the connector payload and return the strict Veilance report."""

        started = time.perf_counter()

        logger.info(
            "Privacy-policy comparison request received "
            "payload_type=%s payload_size=%s",
            type(payload).__name__,
            _format_bytes(
                _json_size(payload)
            ),
        )

        try:
            result = analyze(
                self.tokenizer,
                self.model,
                payload,
                retriever=self.retriever,
                max_new_tokens=self.max_new_tokens,
            )

            logger.info(
                "Privacy-policy comparison request completed "
                "duration=%.3fs findings=%d",
                time.perf_counter()
                - started,
                len(
                    result.get(
                        "findings",
                        [],
                    )
                ),
            )

            return result

        except Exception:
            logger.exception(
                "Privacy-policy comparison request failed "
                "duration=%.3fs",
                time.perf_counter()
                - started,
            )
            raise


# =============================================================================
# Prompt construction
# =============================================================================


def _chat_inputs(
    tokenizer,
    analysis_input: dict,
):
    started = time.perf_counter()

    logger.info(
        "Building model prompt"
    )

    user_prompt_started = (
        time.perf_counter()
    )

    user_prompt = build_user_prompt(
        analysis_input
    )

    logger.info(
        "User prompt built chars=%d duration=%.3fs",
        len(user_prompt),
        time.perf_counter()
        - user_prompt_started,
    )

    logger.debug(
        "System prompt chars=%d user_prompt_chars=%d "
        "analysis_input_size=%s",
        len(SYSTEM_PROMPT),
        len(user_prompt),
        _format_bytes(
            _json_size(
                analysis_input
            )
        ),
    )

    logger.debug(f"=== START RAW SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n=== END RAW SYSTEM PROMPT ===")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:
        logger.debug(
            "Applying tokenizer chat template "
            "enable_thinking=false"
        )

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )

        logger.debug(
            "Tokenizer accepted enable_thinking=false"
        )

    except TypeError:
        logger.warning(
            "Tokenizer chat template does not support "
            "enable_thinking; appending /no_think fallback"
        )

        messages[-1][
            "content"
        ] += "\n/no_think"

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    try:
        input_tokens = int(
            encoded[
                "input_ids"
            ].shape[-1]
        )

    except Exception:
        input_tokens = -1

    logger.info(
        "Model prompt tokenized "
        "input_tokens=%d duration=%.3fs",
        input_tokens,
        time.perf_counter()
        - started,
    )

    if input_tokens > 0:
        try:
            model_max_length = int(
                tokenizer.model_max_length
            )

            logger.debug(
                "Tokenizer context metadata "
                "input_tokens=%d model_max_length=%d "
                "remaining_before_generation=%d",
                input_tokens,
                model_max_length,
                model_max_length
                - input_tokens,
            )

        except Exception:
            pass

    return encoded


# =============================================================================
# Generation
# =============================================================================


def generate_report(
    tokenizer,
    model,
    analysis_input: dict,
    *,
    max_new_tokens: int = 3_200,
) -> dict:
    import torch

    started = time.perf_counter()

    logger.info(
        "Starting Verity report generation "
        "domain=%s max_new_tokens=%d",
        analysis_input.get(
            "domain_url"
        ),
        max_new_tokens,
    )

    policy_document = analysis_input.get(
        "policy_document",
        {},
    )

    logger.info(
        "Generation policy context "
        "found=%s applicable=%s "
        "url=%s sections=%d",
        policy_document.get(
            "found"
        ),
        policy_document.get(
            "applicable"
        ),
        policy_document.get(
            "url"
        )
        or "<none>",
        len(
            policy_document.get(
                "sections",
                [],
            )
        ),
    )

    encoded = _chat_inputs(
        tokenizer,
        analysis_input,
    )

    try:
        input_tokens = int(
            encoded[
                "input_ids"
            ].shape[-1]
        )

    except Exception:
        input_tokens = 0

    logger.debug(
        "Moving prompt tensors to model device=%s "
        "tensor_keys=%s",
        getattr(
            model,
            "device",
            "unknown",
        ),
        list(
            encoded.keys()
        ),
    )

    encoded = {
        key: value.to(
            model.device
        )
        for key, value
        in encoded.items()
    }

    _log_cuda_memory(
        torch,
        "Before generation",
        level=logging.INFO,
    )

    generation_started = (
        time.perf_counter()
    )

    logger.info(
        "Calling model.generate "
        "input_tokens=%d "
        "max_new_tokens=%d "
        "do_sample=false "
        "repetition_penalty=1.02",
        input_tokens,
        max_new_tokens,
    )

    try:
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.02,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    except Exception:
        logger.exception(
            "Model generation failed "
            "input_tokens=%d max_new_tokens=%d "
            "duration=%.3fs",
            input_tokens,
            max_new_tokens,
            time.perf_counter()
            - generation_started,
        )

        _log_cuda_memory(
            torch,
            "Generation failure",
            level=logging.ERROR,
        )

        raise

    generation_duration = (
        time.perf_counter()
        - generation_started
    )

    total_tokens = int(
        output[0].shape[-1]
    )

    generated_tokens = max(
        0,
        total_tokens
        - input_tokens,
    )

    tokens_per_second = (
        generated_tokens
        / generation_duration
        if generation_duration > 0
        else 0.0
    )

    logger.info(
        "Model generation complete "
        "input_tokens=%d output_tokens=%d "
        "total_tokens=%d duration=%.3fs "
        "tokens_per_second=%.2f",
        input_tokens,
        generated_tokens,
        total_tokens,
        generation_duration,
        tokens_per_second,
    )

    if (
        generated_tokens
        >= max_new_tokens
    ):
        logger.warning(
            "Generation reached max_new_tokens limit "
            "generated_tokens=%d limit=%d; "
            "output may have been truncated",
            generated_tokens,
            max_new_tokens,
        )

    _log_cuda_memory(
        torch,
        "After generation",
        level=logging.INFO,
    )

    decode_started = (
        time.perf_counter()
    )

    generated = tokenizer.decode(
        output[
            0,
            input_tokens:
        ],
        skip_special_tokens=True,
    )

    logger.info(
        "Generated tokens decoded "
        "chars=%d duration=%.3fs",
        len(generated),
        time.perf_counter()
        - decode_started,
    )

    logger.debug(
        "Generated output prefix=%r",
        generated[:500],
    )

    parsed = parse_json(
        generated
    )

    logger.info(
        "Finalizing generated report"
    )

    finalized = finalize_report(
        parsed,
        analysis_input,
    )

    logger.info(
        "Verity report generation complete "
        "domain=%s findings=%d "
        "total_duration=%.3fs",
        finalized.get(
            "domain"
        ),
        len(
            finalized.get(
                "findings",
                [],
            )
        ),
        time.perf_counter()
        - started,
    )

    return finalized


# =============================================================================
# Analysis-input preparation
# =============================================================================


def prepare_analysis_input(
    record: dict,
    *,
    domain_url: Optional[str] = None,
    privacy_policy_url: Optional[str] = None,
    use_supplied_policy_document: bool = False,
    retriever: Optional[
        PolicyRetriever
    ] = None,
) -> dict:
    started = time.perf_counter()

    logger.info(
        "Preparing normalized analysis input "
        "domain_override=%s "
        "policy_url_override=%s "
        "use_supplied_policy_document=%s",
        domain_url or "<none>",
        privacy_policy_url or "<none>",
        use_supplied_policy_document,
    )

    logger.debug(
        "Raw input record "
        "type=%s keys=%s size=%s",
        type(record).__name__,
        (
            sorted(
                record.keys()
            )
            if isinstance(
                record,
                dict,
            )
            else "<not-object>"
        ),
        _format_bytes(
            _json_size(record)
        ),
    )

    normalization_started = (
        time.perf_counter()
    )

    try:
        runtime = normalize_runtime_input(
            record,
            domain_url_override=domain_url,
            policy_url_override=privacy_policy_url,
        )

    except Exception:
        logger.exception(
            "Runtime input normalization failed"
        )
        raise

    logger.info(
        "Runtime input normalized "
        "domain=%s privacy_policy_url=%s "
        "duration=%.3fs",
        runtime.get(
            "domain_url"
        ),
        runtime.get(
            "privacy_policy_url"
        )
        or "<none>",
        time.perf_counter()
        - normalization_started,
    )

    logger.debug(
        "Normalized runtime keys=%s size=%s",
        sorted(
            runtime.keys()
        ),
        _format_bytes(
            _json_size(runtime)
        ),
    )

    if use_supplied_policy_document:
        logger.info(
            "Using policy_document supplied in input; "
            "network retrieval is disabled for this analysis"
        )

        supplied_document = runtime.get(
            "supplied_policy_document"
        )

        if not isinstance(
            supplied_document,
            dict,
        ):
            logger.error(
                "Supplied policy-document mode requested "
                "but input contains no valid policy_document"
            )

            raise ValueError(
                "--use-supplied-policy-document "
                "requires policy_document in the input"
            )

        validation_started = (
            time.perf_counter()
        )

        try:
            policy_document = (
                validate_policy_document(
                    supplied_document
                )
            )

        except Exception:
            logger.exception(
                "Supplied policy_document validation failed"
            )
            raise

        logger.info(
            "Supplied policy_document validated "
            "found=%s applicable=%s "
            "url=%s sections=%d duration=%.3fs",
            policy_document.get(
                "found"
            ),
            policy_document.get(
                "applicable"
            ),
            policy_document.get(
                "url"
            )
            or "<none>",
            len(
                policy_document.get(
                    "sections",
                    [],
                )
            ),
            time.perf_counter()
            - validation_started,
        )

    else:
        if retriever is None:
            logger.info(
                "No retriever supplied; creating default PolicyRetriever"
            )

            retriever = PolicyRetriever(
                logger=logging.getLogger(
                    "verity.policy_retrieval"
                ),
                log_level=logging.getLevelName(
                    logger.getEffectiveLevel()
                ),
            )

        logger.info(
            "Starting privacy-policy retrieval "
            "domain=%s supplied_url=%s",
            runtime[
                "domain_url"
            ],
            runtime.get(
                "privacy_policy_url"
            )
            or "<none>",
        )

        retrieval_started = (
            time.perf_counter()
        )

        try:
            policy_document = retriever.retrieve(
                domain_url=runtime[
                    "domain_url"
                ],
                privacy_policy_url=runtime[
                    "privacy_policy_url"
                ],
            )

        except Exception:
            logger.exception(
                "Privacy-policy retrieval raised an exception "
                "domain=%s",
                runtime.get(
                    "domain_url"
                ),
            )
            raise

        logger.info(
            "Privacy-policy retrieval returned "
            "found=%s applicable=%s complete=%s "
            "method=%s url=%s sections=%d "
            "duration=%.3fs",
            policy_document.get(
                "found"
            ),
            policy_document.get(
                "applicable"
            ),
            policy_document.get(
                "complete"
            ),
            policy_document.get(
                "retrieval_method"
            ),
            policy_document.get(
                "url"
            )
            or "<none>",
            len(
                policy_document.get(
                    "sections",
                    [],
                )
            ),
            time.perf_counter()
            - retrieval_started,
        )

        if not policy_document.get(
            "found"
        ):
            logger.warning(
                "No applicable policy was retrieved "
                "attempted_candidates=%s limitations=%s",
                policy_document.get(
                    "attempted_candidates"
                ),
                policy_document.get(
                    "limitations"
                ),
            )

    build_started = (
        time.perf_counter()
    )

    try:
        analysis_input = (
            build_analysis_input(
                runtime,
                policy_document,
            )
        )

    except Exception:
        logger.exception(
            "Building normalized analysis input failed"
        )
        raise

    logger.info(
        "Analysis input built "
        "domain=%s size=%s duration=%.3fs",
        analysis_input.get(
            "domain_url"
        ),
        _format_bytes(
            _json_size(
                analysis_input
            )
        ),
        time.perf_counter()
        - build_started,
    )

    visit = analysis_input.get(
        "visit",
        {},
    )

    logger.info(
        "Analysis visit metadata "
        "observed_at=%s duration_seconds=%s",
        visit.get(
            "observed_at"
        ),
        visit.get(
            "duration_seconds"
        ),
    )

    logger.debug(
        "Analysis input keys=%s",
        sorted(
            analysis_input.keys()
        ),
    )

    logger.info(
        "Analysis input preparation complete "
        "duration=%.3fs",
        time.perf_counter()
        - started,
    )

    return analysis_input


# =============================================================================
# Full analysis
# =============================================================================


def analyze(
    tokenizer,
    model,
    record: dict,
    *,
    retriever: Optional[
        PolicyRetriever
    ] = None,
    use_supplied_policy_document: bool = False,
    max_new_tokens: int = 3_200,
) -> dict:
    started = time.perf_counter()

    logger.info(
        "Starting end-to-end privacy-policy analysis "
        "use_supplied_policy_document=%s "
        "max_new_tokens=%d",
        use_supplied_policy_document,
        max_new_tokens,
    )

    try:
        analysis_input = prepare_analysis_input(
            record,
            use_supplied_policy_document=use_supplied_policy_document,
            retriever=retriever,
        )

        report = generate_report(
            tokenizer,
            model,
            analysis_input,
            max_new_tokens=max_new_tokens,
        )

        logger.info(
            "End-to-end analysis complete "
            "domain=%s findings=%d duration=%.3fs",
            report.get(
                "domain"
            ),
            len(
                report.get(
                    "findings",
                    [],
                )
            ),
            time.perf_counter()
            - started,
        )

        return report

    except Exception:
        logger.exception(
            "End-to-end analysis failed "
            "duration=%.3fs",
            time.perf_counter()
            - started,
        )

        raise


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve an applicable privacy policy with Playwright, compare it "
            "with one Veilance observation, and emit strict JSON."
        )
    )

    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-1.7B",
    )

    parser.add_argument(
        "--adapter",
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--domain-url",
    )

    parser.add_argument(
        "--privacy-policy-url",
    )

    parser.add_argument(
        "--browser",
        choices=[
            "chromium",
            "firefox",
            "webkit",
        ],
        default="chromium",
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--max-policy-chars",
        type=int,
        default=24_000,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3_200,
    )

    parser.add_argument(
        "--no-search",
        dest="search_enabled",
        action="store_false",
        help=(
            "Disable the web-search fallback; direct URL, footer links, "
            "and common paths remain enabled."
        ),
    )

    parser.set_defaults(
        search_enabled=True
    )

    parser.add_argument(
        "--headed",
        action="store_true",
    )

    parser.add_argument(
        "--allow-private-network",
        action="store_true",
        help=(
            "Allow localhost/private targets "
            "for controlled development only."
        ),
    )

    parser.add_argument(
        "--use-supplied-policy-document",
        action="store_true",
        help=(
            "Offline/test mode: trust policy_document from the input "
            "instead of retrieving it."
        ),
    )

    parser.add_argument(
        "--retrieved-policy-output",
        help=(
            "Optional path for the normalized "
            "host-prepared model input."
        ),
    )

    # -------------------------------------------------------------------------
    # Logging arguments
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--log-level",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        default="INFO",
        help=(
            "Logging verbosity. Default: INFO."
        ),
    )

    parser.add_argument(
        "--log-file",
        help=(
            "Optional file to receive the same logs written to stderr."
        ),
    )

    parser.add_argument(
        "--log-format",
        help=(
            "Optional Python logging format string."
        ),
    )

    args = parser.parse_args()

    configure_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        log_format=args.log_format,
    )

    program_started = (
        time.perf_counter()
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "Starting Verity inference CLI"
    )

    logger.info(
        "CLI configuration "
        "base_model=%s adapter=%s input=%s "
        "domain_override=%s policy_override=%s "
        "browser=%s headed=%s timeout_ms=%d "
        "max_policy_chars=%d max_new_tokens=%d "
        "search_enabled=%s allow_private_network=%s "
        "use_supplied_policy_document=%s "
        "retrieved_policy_output=%s "
        "log_level=%s log_file=%s",
        args.base_model,
        args.adapter,
        args.input,
        args.domain_url or "<none>",
        args.privacy_policy_url
        or "<none>",
        args.browser,
        args.headed,
        args.timeout_ms,
        args.max_policy_chars,
        args.max_new_tokens,
        args.search_enabled,
        args.allow_private_network,
        args.use_supplied_policy_document,
        args.retrieved_policy_output
        or "<none>",
        args.log_level,
        args.log_file
        or "<none>",
    )

    try:
        # ---------------------------------------------------------------------
        # Input file
        # ---------------------------------------------------------------------

        input_path = Path(
            args.input
        )

        logger.info(
            "Reading input file path=%s",
            input_path,
        )

        file_started = (
            time.perf_counter()
        )

        raw_input = input_path.read_text(
            encoding="utf-8"
        )

        logger.info(
            "Input file read bytes=%s chars=%d duration=%.3fs",
            _format_bytes(
                len(
                    raw_input.encode(
                        "utf-8"
                    )
                )
            ),
            len(raw_input),
            time.perf_counter()
            - file_started,
        )

        parse_started = (
            time.perf_counter()
        )

        try:
            record = json.loads(
                raw_input
            )

        except json.JSONDecodeError as exc:
            logger.error(
                "Input JSON decoding failed "
                "line=%d column=%d position=%d error=%s",
                exc.lineno,
                exc.colno,
                exc.pos,
                exc.msg,
            )

            raise

        if not isinstance(
            record,
            dict,
        ):
            logger.error(
                "Input JSON root must be an object; "
                "received type=%s",
                type(record).__name__,
            )

            raise ValueError(
                "input JSON must contain one object"
            )

        logger.info(
            "Input JSON parsed "
            "keys=%s duration=%.3fs",
            sorted(
                record.keys()
            ),
            time.perf_counter()
            - parse_started,
        )

        # ---------------------------------------------------------------------
        # Policy retriever
        # ---------------------------------------------------------------------

        retriever = None

        if not args.use_supplied_policy_document:
            logger.info(
                "Configuring PolicyRetriever"
            )

            retriever = PolicyRetriever(
                timeout_ms=args.timeout_ms,
                max_policy_chars=args.max_policy_chars,
                search_enabled=args.search_enabled,
                browser_name=args.browser,
                headless=not args.headed,
                allow_private_network=args.allow_private_network,

                # Use the same logging hierarchy.
                log_level=args.log_level,
                logger=logging.getLogger(
                    "verity.policy_retrieval"
                ),
            )

            logger.info(
                "PolicyRetriever configured "
                "browser=%s headless=%s "
                "timeout_ms=%d search_enabled=%s "
                "max_policy_chars=%d",
                args.browser,
                not args.headed,
                args.timeout_ms,
                args.search_enabled,
                args.max_policy_chars,
            )

        else:
            logger.info(
                "Policy retrieval disabled because "
                "--use-supplied-policy-document was specified"
            )

        # ---------------------------------------------------------------------
        # Analysis input
        # ---------------------------------------------------------------------

        analysis_input = prepare_analysis_input(
            record,
            domain_url=args.domain_url,
            privacy_policy_url=args.privacy_policy_url,
            use_supplied_policy_document=args.use_supplied_policy_document,
            retriever=retriever,
        )

        # ---------------------------------------------------------------------
        # Optional host-prepared input output
        # ---------------------------------------------------------------------

        if args.retrieved_policy_output:
            output_path = Path(
                args.retrieved_policy_output
            )

            logger.info(
                "Writing normalized analysis input path=%s",
                output_path,
            )

            if (
                output_path.parent
                != Path(".")
            ):
                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            serialized_analysis = (
                json.dumps(
                    analysis_input,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )

            output_path.write_text(
                serialized_analysis,
                encoding="utf-8",
            )

            logger.info(
                "Normalized analysis input written "
                "path=%s bytes=%s",
                output_path,
                _format_bytes(
                    len(
                        serialized_analysis.encode(
                            "utf-8"
                        )
                    )
                ),
            )

        # ---------------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------------

        tokenizer, model = load_model(
            args.base_model,
            args.adapter,
        )

        # ---------------------------------------------------------------------
        # Inference
        # ---------------------------------------------------------------------

        report = generate_report(
            tokenizer,
            model,
            analysis_input,
            max_new_tokens=args.max_new_tokens,
        )

        # ---------------------------------------------------------------------
        # Final stdout JSON
        # ---------------------------------------------------------------------

        logger.info(
            "Serializing final report for stdout"
        )

        serialized_report = json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )

        logger.info(
            "Final report ready "
            "domain=%s findings=%d "
            "output_chars=%d output_bytes=%s",
            report.get(
                "domain"
            ),
            len(
                report.get(
                    "findings",
                    [],
                )
            ),
            len(serialized_report),
            _format_bytes(
                len(
                    serialized_report.encode(
                        "utf-8"
                    )
                )
            ),
        )

        # IMPORTANT:
        # stdout stays reserved for strict report JSON.
        print(
            serialized_report
        )

        logger.info(
            "Verity inference CLI completed successfully "
            "total_duration=%.3fs",
            time.perf_counter()
            - program_started,
        )

        logger.info(
            "============================================================"
        )

    except KeyboardInterrupt:
        logger.warning(
            "Verity inference interrupted by user "
            "duration=%.3fs",
            time.perf_counter()
            - program_started,
        )

        logger.info(
            "============================================================"
        )

        raise

    except Exception:
        logger.exception(
            "Fatal Verity inference error "
            "duration=%.3fs",
            time.perf_counter()
            - program_started,
        )

        logger.info(
            "============================================================"
        )

        raise


if __name__ == "__main__":
    main()
