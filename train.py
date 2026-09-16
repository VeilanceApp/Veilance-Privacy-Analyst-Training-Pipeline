import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def render_non_thinking_example(example: dict, tokenizer) -> dict:
    prompt_messages = example["prompt"]
    completion_messages = example["completion"]
    if not isinstance(prompt_messages, list) or not isinstance(completion_messages, list):
        raise ValueError("prepared dataset prompt/completion must be message arrays")
    if len(completion_messages) != 1 or completion_messages[0].get("role") != "assistant":
        raise ValueError("each completion must contain one assistant message")
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_messages = [dict(message) for message in prompt_messages]
        prompt_messages[-1]["content"] += "\n/no_think"
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    completion_text = completion_messages[0]["content"] + (tokenizer.eos_token or "")
    return {"prompt": prompt_text, "completion": completion_text}


def validate_lengths(dataset, tokenizer, max_length: int, split_name: str) -> None:
    too_long = []
    for index, example in enumerate(dataset):
        token_count = len(
            tokenizer(
                example["prompt"] + example["completion"],
                add_special_tokens=False,
            )["input_ids"]
        )
        if token_count > max_length:
            too_long.append((index, token_count))
            if len(too_long) >= 10:
                break
    if too_long:
        details = ", ".join(f"row {index}: {count}" for index, count in too_long)
        raise ValueError(
            f"{split_name} contains examples longer than --max-length={max_length} ({details}). "
            "Reduce policy sections or raise max length; silent truncation would corrupt JSON completions."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    train_path = Path(args.train).resolve()
    eval_path = Path(args.eval).resolve()
    done_file = Path(args.output).resolve() / "DONE"
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and done_file.exists():
        done_file.unlink()

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(
        "json", data_files=str(train_path), split="train"
    )
    eval_dataset = load_dataset(
        "json", data_files=str(eval_path), split="train"
    )
    keep_columns = {"prompt", "completion"}
    train_dataset = train_dataset.map(
        lambda example: render_non_thinking_example(example, tokenizer),
        remove_columns=[
            column for column in train_dataset.column_names if column not in keep_columns
        ],
        desc="Render non-thinking training prompts",
    )
    eval_dataset = eval_dataset.map(
        lambda example: render_non_thinking_example(example, tokenizer),
        remove_columns=[
            column for column in eval_dataset.column_names if column not in keep_columns
        ],
        desc="Render non-thinking evaluation prompts",
    )
    validate_lengths(train_dataset, tokenizer, args.max_length, "training split")
    validate_lengths(eval_dataset, tokenizer, args.max_length, "validation split")

    quantization = BitsAndBytesConfig(
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
    config = SFTConfig(
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        model_init_kwargs={
            "quantization_config": quantization,
            "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},
            "torch_dtype": dtype,
        },
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if trainer.is_world_process_zero():
        done_file.parent.mkdir(parents=True, exist_ok=True)
        done_file.write_text("DONE\n", encoding="utf-8")
        metadata = {
            "base_model": args.model,
            "max_length": args.max_length,
            "thinking_enabled": False,
            "completion_only_loss": True,
        }
        (done_file.parent / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Training complete. Created {done_file}")


if __name__ == "__main__":
    main()
