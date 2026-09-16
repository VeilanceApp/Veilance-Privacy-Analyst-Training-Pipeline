import argparse
import json
import re
import sys
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


def load_model(base: str, adapter: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the configured 4-bit inference path")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter)
    model.eval()
    return tokenizer, model


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.S)
    if text.startswith("```") or text.endswith("```"):
        raise ValueError("model returned a Markdown code fence instead of raw JSON")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    return value


def _append_unique(values: list, additions: list[str]) -> list:
    output = [item for item in values if isinstance(item, str)]
    for addition in additions:
        if addition and addition not in output:
            output.append(addition)
    return output


def _validate_policy_grounding(report: dict, policy_document: dict) -> None:
    """Reject section citations that are not present in retrieved policy text."""

    if not (policy_document["found"] and policy_document["applicable"]):
        return
    headings = {
        section["heading"].strip().casefold()
        for section in policy_document["sections"]
        if section["heading"].strip()
    }
    disclosure_statuses = {
        "explicitly_disclosed",
        "broadly_disclosed",
        "implicitly_disclosed",
        "contradicted",
    }
    for index, finding in enumerate(report.get("findings", [])):
        policy = finding.get("policy") if isinstance(finding, dict) else None
        if not isinstance(policy, dict):
            continue
        status = policy.get("status")
        evidence = policy.get("evidence")
        section = policy.get("section")
        if status in disclosure_statuses and (
            not isinstance(evidence, str) or not evidence.strip()
        ):
            raise ValueError(
                f"findings/{index}/policy/evidence is required for {status}"
            )
        if isinstance(section, str) and section.strip():
            cited = [item.strip().casefold() for item in section.split(";") if item.strip()]
            if not cited or any(item not in headings for item in cited):
                raise ValueError(
                    f"findings/{index}/policy/section cites a section that was not extracted by Playwright"
                )
        elif status in disclosure_statuses:
            raise ValueError(
                f"findings/{index}/policy/section is required for {status}"
            )


def finalize_report(report: dict, analysis_input: dict) -> dict:
    """Apply authoritative host fields and deterministic derived values."""

    policy_document = analysis_input["policy_document"]
    visit = analysis_input["visit"]
    report["domain"] = analysis_input["domain_url"]
    policy_available = policy_document["found"] and policy_document["applicable"]
    report["privacy_policy"] = {
        "url": policy_document["url"] if policy_available else "",
        "found": policy_available,
        "applicable": policy_available,
    }
    report["visit"] = {
        "observed_at": visit.get("observed_at"),
        "duration_seconds": visit["duration_seconds"],
    }

    findings = report.get("findings")
    if not isinstance(findings, list):
        raise ValueError("model output findings must be an array")

    if not policy_available:
        for finding in findings:
            if not isinstance(finding, dict) or not isinstance(
                finding.get("policy"), dict
            ):
                continue
            finding["policy"] = {
                "status": "unknown",
                "evidence": "",
                "section": "",
            }
            finding["comparison"] = "indeterminate"
            finding["severity"] = "informational"
            if isinstance(finding.get("confidence"), (int, float)) and not isinstance(
                finding.get("confidence"), bool
            ):
                finding["confidence"] = min(float(finding["confidence"]), 0.5)
            finding["explanation"] = (
                "No applicable privacy policy was available, so this behavior "
                "cannot be reliably compared with policy disclosure."
            )

    analysis = report.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("model output analysis must be an object")
    if not policy_available:
        observed_findings = sum(
            isinstance(finding, dict)
            and isinstance(finding.get("telemetry"), dict)
            and finding["telemetry"].get("status") == "observed"
            for finding in findings
        )
        analysis["summary"] = (
            f"Veilance identified {observed_findings} grouped observed behaviors "
            "during this visit, but no applicable privacy policy was available, "
            "so the comparisons are indeterminate."
        )
    analysis["counts"] = recompute_counts(findings)
    confidences = [
        finding.get("confidence")
        for finding in findings
        if isinstance(finding, dict)
        and isinstance(finding.get("confidence"), (int, float))
        and not isinstance(finding.get("confidence"), bool)
    ]
    analysis["overall_confidence"] = (
        round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    )

    limitations = report.get("important_limitations")
    if not isinstance(limitations, list):
        raise ValueError("model output important_limitations must be an array")
    report["important_limitations"] = _append_unique(
        limitations,
        policy_document.get("limitations", []),
    )
    _validate_policy_grounding(report, policy_document)
    return validate_report(report)


class PrivacyPolicyComparisonEngine:
    """Reusable local equivalent of the existing ChatGPT connector call."""

    def __init__(
        self,
        *,
        adapter: str,
        base_model: str = "Qwen/Qwen3-1.7B",
        retriever: Optional[PolicyRetriever] = None,
        max_new_tokens: int = 3_200,
    ):
        self.tokenizer, self.model = load_model(base_model, adapter)
        self.retriever = retriever or PolicyRetriever()
        self.max_new_tokens = max_new_tokens

    def privacy_policy_comparison(self, payload: dict) -> dict:
        """Accept the connector payload and return the strict Veilance report."""

        return analyze(
            self.tokenizer,
            self.model,
            payload,
            retriever=self.retriever,
            max_new_tokens=self.max_new_tokens,
        )


def _chat_inputs(tokenizer, analysis_input: dict):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(analysis_input)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        messages[-1]["content"] += "\n/no_think"
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )


def generate_report(
    tokenizer,
    model,
    analysis_input: dict,
    *,
    max_new_tokens: int = 3_200,
) -> dict:
    import torch

    encoded = _chat_inputs(tokenizer, analysis_input)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.02,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output[0, encoded["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    return finalize_report(parse_json(generated), analysis_input)


def prepare_analysis_input(
    record: dict,
    *,
    domain_url: Optional[str] = None,
    privacy_policy_url: Optional[str] = None,
    use_supplied_policy_document: bool = False,
    retriever: Optional[PolicyRetriever] = None,
) -> dict:
    runtime = normalize_runtime_input(
        record,
        domain_url_override=domain_url,
        policy_url_override=privacy_policy_url,
    )
    if use_supplied_policy_document:
        if not isinstance(runtime["supplied_policy_document"], dict):
            raise ValueError(
                "--use-supplied-policy-document requires policy_document in the input"
            )
        policy_document = validate_policy_document(
            runtime["supplied_policy_document"]
        )
    else:
        retriever = retriever or PolicyRetriever()
        policy_document = retriever.retrieve(
            domain_url=runtime["domain_url"],
            privacy_policy_url=runtime["privacy_policy_url"],
        )
    return build_analysis_input(runtime, policy_document)


def analyze(
    tokenizer,
    model,
    record: dict,
    *,
    retriever: Optional[PolicyRetriever] = None,
    use_supplied_policy_document: bool = False,
    max_new_tokens: int = 3_200,
) -> dict:
    analysis_input = prepare_analysis_input(
        record,
        use_supplied_policy_document=use_supplied_policy_document,
        retriever=retriever,
    )
    return generate_report(
        tokenizer,
        model,
        analysis_input,
        max_new_tokens=max_new_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve an applicable privacy policy with Playwright, compare it "
            "with one Veilance observation, and emit strict JSON."
        )
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--domain-url")
    parser.add_argument("--privacy-policy-url")
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], default="chromium")
    parser.add_argument("--timeout-ms", type=int, default=20_000)
    parser.add_argument("--max-policy-chars", type=int, default=24_000)
    parser.add_argument("--max-new-tokens", type=int, default=3_200)
    parser.add_argument(
        "--no-search",
        dest="search_enabled",
        action="store_false",
        help="Disable the web-search fallback; direct URL, footer links, and common paths remain enabled.",
    )
    parser.set_defaults(search_enabled=True)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--allow-private-network",
        action="store_true",
        help="Allow localhost/private targets for controlled development only.",
    )
    parser.add_argument(
        "--use-supplied-policy-document",
        action="store_true",
        help="Offline/test mode: trust policy_document from the input instead of retrieving it.",
    )
    parser.add_argument(
        "--retrieved-policy-output",
        help="Optional path for the normalized host-prepared model input.",
    )
    args = parser.parse_args()

    record = json.loads(Path(args.input).read_text(encoding="utf-8"))
    retriever = None
    if not args.use_supplied_policy_document:
        retriever = PolicyRetriever(
            timeout_ms=args.timeout_ms,
            max_policy_chars=args.max_policy_chars,
            search_enabled=args.search_enabled,
            browser_name=args.browser,
            headless=not args.headed,
            allow_private_network=args.allow_private_network,
        )
        print("Retrieving and verifying the applicable privacy policy...", file=sys.stderr)

    analysis_input = prepare_analysis_input(
        record,
        domain_url=args.domain_url,
        privacy_policy_url=args.privacy_policy_url,
        use_supplied_policy_document=args.use_supplied_policy_document,
        retriever=retriever,
    )
    if args.retrieved_policy_output:
        Path(args.retrieved_policy_output).write_text(
            json.dumps(analysis_input, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    tokenizer, model = load_model(args.base_model, args.adapter)
    report = generate_report(
        tokenizer,
        model,
        analysis_input,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
