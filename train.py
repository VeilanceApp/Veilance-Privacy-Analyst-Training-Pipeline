import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16)

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    train_path = Path(args.train).resolve()
    done_file = train_path.parent / "DONE"

    rank = int(os.environ.get("RANK", "0"))

    if rank == 0 and done_file.exists():
        done_file.unlink()

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    cfg = SFTConfig(
        output_dir=args.output,

        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,

        weight_decay=0.01,
        lr_scheduler_type="cosine",

        logging_steps=10,

        eval_strategy="steps",
        eval_steps=50,

        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,

        gradient_checkpointing=True,

        bf16=bf16,
        fp16=not bf16,

        report_to="none",

        model_init_kwargs={
            "quantization_config": q,
            "device_map": {
                "": int(os.environ.get("LOCAL_RANK", "0"))
            },
            "torch_dtype": dtype,
        },
    )

    train_dataset = load_dataset(
        "json",
        data_files=args.train,
        split="train",
    )
    eval_dataset = load_dataset(
        "json",
        data_files=args.eval,
        split="train",
    )

    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tok,
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model(args.output)

    if trainer.is_world_process_zero():
        tok.save_pretrained(args.output)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if trainer.is_world_process_zero():
        done_file.write_text("DONE\n", encoding="utf-8")
        print(f"\nTraining complete.")
        print(f"Created completion marker: {done_file}")


if __name__ == "__main__":
    main()