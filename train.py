import argparse
import json
import shutil

from pathlib import Path
from datetime import timedelta

import torch

from accelerate import PartialState
from accelerate.utils import InitProcessGroupKwargs
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


def render_non_thinking_example(example: dict, tokenizer) -> dict:
    prompt_messages = example["prompt"]
    completion_messages = example["completion"]

    if not isinstance(prompt_messages, list):
        raise ValueError("prepared dataset prompt must be a message array")

    if not isinstance(completion_messages, list):
        raise ValueError("prepared dataset completion must be a message array")

    if (
        len(completion_messages) != 1
        or completion_messages[0].get("role") != "assistant"
    ):
        raise ValueError(
            "each completion must contain exactly one assistant message"
        )

    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:
        prompt_messages = [
            dict(message)
            for message in prompt_messages
        ]

        prompt_messages[-1]["content"] += "\n/no_think"

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    completion_text = (
        completion_messages[0]["content"]
        + (tokenizer.eos_token or "")
    )

    return {
        "prompt": prompt_text,
        "completion": completion_text,
    }


def validate_lengths(
    dataset,
    tokenizer,
    max_length: int,
    split_name: str,
    num_proc: int,
) -> None:

    print(
        f"Validating token lengths for {split_name}: "
        f"{len(dataset):,} examples using {num_proc} processes"
    )

    def calculate_lengths(batch):
        texts = [
            prompt + completion
            for prompt, completion in zip(
                batch["prompt"],
                batch["completion"],
            )
        ]

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )

        return {
            "_token_length": [
                len(input_ids)
                for input_ids in encoded["input_ids"]
            ]
        }

    checked = dataset.map(
        calculate_lengths,
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        desc=f"Tokenize {split_name}",
    )

    too_long = []

    for index, token_count in enumerate(
        checked["_token_length"]
    ):
        if token_count > max_length:
            too_long.append(
                (index, token_count)
            )

            if len(too_long) >= 10:
                break

    if too_long:
        details = ", ".join(
            f"row {index}: {count}"
            for index, count in too_long
        )

        raise ValueError(
            f"{split_name} contains examples longer than "
            f"--max-length={max_length}: {details}. "
            "Reduce policy sections or raise max length. "
            "Silent truncation could corrupt JSON completions."
        )

    print(
        f"{split_name}: all {len(dataset):,} examples "
        f"fit within {max_length:,} tokens"
    )


def preprocess_datasets(
    train_path: Path,
    eval_path: Path,
    train_cache: Path,
    eval_cache: Path,
    tokenizer,
    max_length: int,
    num_proc: int,
) -> None:

    print("Loading raw training dataset...")

    train_dataset = load_dataset(
        "json",
        data_files=str(train_path),
        split="train",
    )

    print("Loading raw evaluation dataset...")

    eval_dataset = load_dataset(
        "json",
        data_files=str(eval_path),
        split="train",
    )

    print(
        f"Raw dataset sizes: "
        f"train={len(train_dataset):,}, "
        f"eval={len(eval_dataset):,}"
    )

    keep_columns = {
        "prompt",
        "completion",
    }

    train_remove_columns = [
        column
        for column in train_dataset.column_names
        if column not in keep_columns
    ]

    eval_remove_columns = [
        column
        for column in eval_dataset.column_names
        if column not in keep_columns
    ]

    print(
        f"Rendering training prompts with "
        f"{num_proc} processes..."
    )

    train_dataset = train_dataset.map(
        render_non_thinking_example,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        remove_columns=train_remove_columns,
        num_proc=num_proc,
        desc="Render non-thinking training prompts",
    )

    print(
        f"Rendering evaluation prompts with "
        f"{num_proc} processes..."
    )

    eval_dataset = eval_dataset.map(
        render_non_thinking_example,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        remove_columns=eval_remove_columns,
        num_proc=num_proc,
        desc="Render non-thinking evaluation prompts",
    )

    validate_lengths(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        split_name="training split",
        num_proc=num_proc,
    )

    validate_lengths(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        split_name="validation split",
        num_proc=num_proc,
    )

    if train_cache.exists():
        shutil.rmtree(train_cache)

    if eval_cache.exists():
        shutil.rmtree(eval_cache)

    train_cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Saving training cache to "
        f"{train_cache}"
    )

    train_dataset.save_to_disk(
        str(train_cache),
        num_proc=num_proc,
    )

    print(
        f"Saving evaluation cache to "
        f"{eval_cache}"
    )

    eval_dataset.save_to_disk(
        str(eval_cache),
        num_proc=num_proc,
    )

    print("Dataset preprocessing complete.")


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
    )

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--eval",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    # One epoch is intentional.
    parser.add_argument(
        "--epochs",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=4096,
        help=(
            "Number of validation examples used during "
            "training. Full validation is still run at end."
        ),
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--skip-final-full-eval",
        action="store_true",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU required"
        )

    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(minutes=60),
    ).to_kwargs()

    state = PartialState(
        **process_group_kwargs
    )

    rank = state.process_index
    local_rank = state.local_process_index

    train_path = Path(
        args.train
    ).resolve()

    eval_path = Path(
        args.eval
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    done_file = (
        output_path
        / "DONE"
    )

    cache_root = (
        output_path.parent
        / f".{output_path.name}-preprocessed"
    )

    train_cache = (
        cache_root
        / "train"
    )

    eval_cache = (
        cache_root
        / "eval"
    )

    if state.is_main_process:

        print("=" * 70)
        print("Verity training")
        print("=" * 70)

        print(
            f"Model:               "
            f"{args.model}"
        )

        print(
            f"Train file:          "
            f"{train_path}"
        )

        print(
            f"Eval file:           "
            f"{eval_path}"
        )

        print(
            f"Output:              "
            f"{output_path}"
        )

        print(
            f"World size:          "
            f"{state.num_processes}"
        )

        print(
            f"Epochs:              "
            f"{args.epochs}"
        )

        print(
            f"Preprocess workers:  "
            f"{args.preprocess_workers}"
        )

        print(
            f"Dataset workers:     "
            f"{args.dataset_workers}"
        )

        print(
            f"DataLoader workers:  "
            f"{args.dataloader_workers}"
        )

        print(
            f"Max length:          "
            f"{args.max_length}"
        )

        print(
            f"Training eval size:  "
            f"{args.eval_samples}"
        )

        print(
            f"Evaluation interval: "
            f"{args.eval_steps}"
        )

        print("=" * 70)

        if done_file.exists():
            done_file.unlink()

    bf16 = torch.cuda.is_bf16_supported()

    dtype = (
        torch.bfloat16
        if bf16
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    cache_exists = (
        train_cache.exists()
        and eval_cache.exists()
    )

    if state.is_main_process:

        if (
            args.rebuild_cache
            or not cache_exists
        ):
            print(
                "Building preprocessed dataset "
                "cache on rank 0..."
            )

            preprocess_datasets(
                train_path=train_path,
                eval_path=eval_path,
                train_cache=train_cache,
                eval_cache=eval_cache,
                tokenizer=tokenizer,
                max_length=args.max_length,
                num_proc=args.preprocess_workers,
            )

        else:
            print(
                "Using existing preprocessed "
                "dataset cache."
            )

    state.wait_for_everyone()

    print(
        f"[rank {rank}] loading "
        f"processed datasets"
    )

    train_dataset = load_from_disk(
        str(train_cache)
    )

    full_eval_dataset = load_from_disk(
        str(eval_cache)
    )

    #
    # Use a deterministic representative subset
    # for frequent training-time evaluation.
    #
    eval_sample_count = min(
        args.eval_samples,
        len(full_eval_dataset),
    )

    eval_dataset = (
        full_eval_dataset
        .shuffle(seed=args.seed)
        .select(
            range(eval_sample_count)
        )
    )

    if state.is_main_process:

        print(
            f"Processed training size: "
            f"{len(train_dataset):,}"
        )

        print(
            f"Full validation size:   "
            f"{len(full_eval_dataset):,}"
        )

        print(
            f"Training validation:    "
            f"{len(eval_dataset):,}"
        )

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
        output_dir=str(
            output_path
        ),

        num_train_epochs=args.epochs,

        learning_rate=args.learning_rate,

        weight_decay=0.01,

        lr_scheduler_type="cosine",

        per_device_train_batch_size=1,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            args.grad_accum
        ),

        dataset_num_proc=(
            args.dataset_workers
        ),

        dataloader_num_workers=(
            args.dataloader_workers
        ),

        dataloader_pin_memory=True,

        #
        # Important for your DDP warning.
        #
        ddp_find_unused_parameters=False,

        ddp_timeout=3600,

        logging_steps=10,

        #
        # Do not evaluate every 50 steps.
        #
        eval_strategy="steps",
        eval_steps=args.eval_steps,

        #
        # Keep checkpoints synchronized with eval.
        #
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        max_length=args.max_length,

        #
        # Leave packing disabled because the
        # completion boundaries matter here.
        #
        packing=False,

        completion_only_loss=True,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        bf16=bf16,

        fp16=not bf16,

        report_to="none",

        seed=args.seed,

        data_seed=args.seed,

        model_init_kwargs={
            "quantization_config": (
                quantization
            ),
            "device_map": {
                "": local_rank
            },
            "torch_dtype": dtype,
        },
    )

    callbacks = []

    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=(
                    args.early_stopping_patience
                ),
                early_stopping_threshold=0.001,
            )
        )

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=callbacks,
    )

    if state.is_main_process:
        print("Starting training...")

    trainer.train()

    #
    # Because load_best_model_at_end=True,
    # trainer.model now contains the best
    # validation checkpoint.
    #
    trainer.save_model(
        str(output_path)
    )

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(
            str(output_path)
        )

    #
    # Run the expensive complete validation
    # dataset exactly once.
    #
    final_metrics = None

    if not args.skip_final_full_eval:

        if state.is_main_process:
            print(
                "Running final evaluation "
                f"against all "
                f"{len(full_eval_dataset):,} "
                "validation examples..."
            )

        final_metrics = trainer.evaluate(
            eval_dataset=full_eval_dataset,
            metric_key_prefix="final_eval",
        )

        if trainer.is_world_process_zero():

            (
                output_path
                / "final_eval_metrics.json"
            ).write_text(
                json.dumps(
                    final_metrics,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()

    if trainer.is_world_process_zero():

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        done_file.write_text(
            "DONE\n",
            encoding="utf-8",
        )

        metadata = {
            "base_model": args.model,
            "max_length": args.max_length,
            "thinking_enabled": False,
            "completion_only_loss": True,
            "epochs": args.epochs,
            "learning_rate": (
                args.learning_rate
            ),
            "lora_r": args.lora_r,
            "gradient_accumulation_steps": (
                args.grad_accum
            ),
            "world_size": (
                state.num_processes
            ),
            "bf16": bf16,
            "training_eval_samples": (
                eval_sample_count
            ),
            "full_eval_samples": (
                len(full_eval_dataset)
            ),
            "eval_steps": (
                args.eval_steps
            ),
            "early_stopping_patience": (
                args.early_stopping_patience
            ),
            "seed": args.seed,
        }

        (
            output_path
            / "training_metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"Training complete. "
            f"Created {done_file}"
        )


if __name__ == "__main__":
    main()