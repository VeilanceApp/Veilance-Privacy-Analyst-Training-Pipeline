# Veilance Privacy Analyst - Qwen3-1.7B

## Model

Default base model:

    Qwen/Qwen3-1.7B

Training method: 4-bit NF4 QLoRA.

## Target output

The assistant output matches the current Veilance Privacy Analyst report shape:

- `domain`
- `privacy_policy`
- `visit`
- `analysis`
- `findings`
- `important_limitations`

See `examples/virustotal_expected_report.json`.

## Raw training row format

Each JSONL row contains three top-level objects:

```json
{
  "telemetry": { "... exact veilance.telemetry-snapshot.v2 ...": true },
  "policy_document": {
    "url": "https://example.com/privacy",
    "found": true,
    "applicable": true,
    "sections": [
      {"heading": "Information We Collect", "text": "..."}
    ]
  },
  "expected": { "... exact analyst report ...": true }
}
```

`policy_document.sections` is backend/browser-retrieved policy text. It is model input only. The `expected.privacy_policy` object remains exactly the compact report object used by Veilance.

## Extract a real extension upload

If you save the uploaded `telemetry.bin` from the API:

```bash
python extract_telemetry.py telemetry.bin --output observations.jsonl
```

The extractor accepts either:

- gzip `telemetry.bin`
- uncompressed batch JSON
- a single snapshot JSON

It does not modify the observation payload.

## Prepare training data

```bash
python prepare_dataset.py   --input data/raw.jsonl   --output-dir data/processed
```

Splits are domain-aware.

## Train

```bash
python train.py   --train data/processed/train.jsonl   --eval data/processed/validation.jsonl   --output models/veilance-qwen3-1.7b   --epochs 3   --max-length 6144   --learning-rate 1e-4   --lora-r 32
```

If policies are already reduced to relevant sections, 4096 tokens is usually a better first experiment.

## Inference

```bash
python infer.py   --adapter models/veilance-qwen3-1.7b   --input examples/inference_input.json
```

## Evaluate

```bash
python evaluate.py   --adapter models/veilance-qwen3-1.7b   --test data/processed/test.jsonl
```

The evaluator measures exact report-schema validity, finding comparison accuracy, macro F1, missing/extra findings, and false accusation rate.

## Important

Do not teach the model that every API read implies tracking or collection. The report should preserve Veilance's current conservative language:

- observed behavior is not proof of purpose
- third-party contact is not proof of transmitted personal data
- browser storage may predate the visit
- API access is not automatically fingerprinting
- short visits cannot prove absence of behavior
