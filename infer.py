import argparse, json, re, torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompt import SYSTEM_PROMPT, build_user_prompt
from schema import validate_report
from telemetry import validate_snapshot_shape


def load_model(base,adapter):
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(adapter,use_fast=True)
    model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base,quantization_config=q,torch_dtype=dtype,device_map='auto'),adapter)
    model.eval(); return tok,model


def parse_json(text):
    text=re.sub(r'^<think>.*?</think>\s*','',text.strip(),flags=re.S)
    try:return json.loads(text)
    except: return json.loads(text[text.find('{'):text.rfind('}')+1])

def normalize_report(report):
    findings = report.get("findings", [])

    counts = {
        "matched": 0,
        "partially_matched": 0,
        "policy_only": 0,
        "observed_only": 0,
        "possible_contradictions": 0,
        "indeterminate": 0,
    }

    mapping = {
        "matched": "matched",
        "partially_matched": "partially_matched",
        "policy_only": "policy_only",
        "observed_only": "observed_only",
        "possible_contradiction": "possible_contradictions",
        "indeterminate": "indeterminate",
    }

    confidences = []

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        comparison = finding.get("comparison")

        field = mapping.get(comparison)

        if field:
            counts[field] += 1

        confidence = finding.get("confidence")

        if isinstance(confidence, float):
            confidences.append(confidence)

    analysis = report.get("analysis")

    if isinstance(analysis, dict):
        analysis["counts"] = counts
        if confidences:
            analysis["overall_confidence"] = round(
                sum(confidences) / len(confidences),
                2,
            )
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-model',default='Qwen/Qwen3-1.7B'); ap.add_argument('--adapter',required=True); ap.add_argument('--input',required=True); args=ap.parse_args()
    record=json.load(open(args.input,encoding='utf-8')); tok,model=load_model(args.base_model,args.adapter); print(json.dumps(analyze(tok,model,record),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
