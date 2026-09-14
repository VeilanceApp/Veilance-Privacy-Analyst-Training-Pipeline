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

@torch.inference_mode()
def analyze(tok,model,record,max_new_tokens=2600):
    validate_snapshot_shape(record['telemetry'])
    msgs=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':build_user_prompt(record)}]
    try: enc=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)
    except TypeError: enc=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
    enc={k:v.to(model.device) for k,v in enc.items()}
    out=model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,repetition_penalty=1.02,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    report=parse_json(tok.decode(out[0,enc['input_ids'].shape[-1]:],skip_special_tokens=True)); validate_report(report); return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-model',default='Qwen/Qwen3-1.7B'); ap.add_argument('--adapter',required=True); ap.add_argument('--input',required=True); args=ap.parse_args()
    record=json.load(open(args.input,encoding='utf-8')); tok,model=load_model(args.base_model,args.adapter); print(json.dumps(analyze(tok,model,record),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
