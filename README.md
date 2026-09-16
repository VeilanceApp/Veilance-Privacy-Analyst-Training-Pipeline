# Veilance Privacy Analyst — Qwen3-1.7B

This project fine-tunes and runs a local Qwen3-1.7B privacy-policy comparison engine. It does not call ChatGPT or the OpenAI API.

At inference time the host:

1. accepts the same normalized telemetry object currently passed to `chatgpt.privacy_policy_comparison(...)`;
2. uses Playwright to retrieve and verify the supplied privacy-policy URL;
3. if needed, discovers a replacement through first-party legal links, common policy paths, and a web-search fallback;
4. extracts plain policy sections and passes those sections plus telemetry to the local model;
5. validates and returns one strict Veilance report object.

The model never controls the browser. Policy pages are treated as untrusted evidence, not instructions.

## Requirements

- Python 3.10 or newer
- a CUDA GPU for the configured 4-bit QLoRA training and inference path
- network access from the inference host to public websites
- a Playwright browser installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Runtime input contract

`infer.py` accepts the connector payload exactly as it is already sent to ChatGPT:

```json
{
  "domain_url": "https://x.com",
  "privacy_policy_url": "https://x.com/en/privacy",
  "visit": {
    "snapshot_id": "snapshot-id",
    "observed_at": 1788872953702,
    "duration_seconds": 1,
    "extension_version": "0.8"
  },
  "seen_behavior": {
    "observations": {
      "totalRequests": 29,
      "firstPartyRequests": 6,
      "thirdPartyRequests": 23
    }
  },
  "thirdPartyHosts": [],
  "trackers": [],
  "signals": [],
  "page": {
    "scriptCount": 0,
    "thirdPartyScriptCount": 0,
    "iframeCount": 0,
    "thirdPartyIframeCount": 0,
    "accessibleCookieCount": 0,
    "localStorageKeyCount": 0,
    "sessionStorageKeyCount": 0,
    "indexedDbCount": 0,
    "cacheCount": 0,
    "serviceWorkerControlled": false
  },
  "detections": []
}
```

See `examples/inference_input.json` for a populated example.

The runtime also accepts:

- a raw `veilance.telemetry-snapshot.v2` object;
- an upload envelope containing the snapshot at `payload`;
- a training-style object containing the snapshot at `telemetry`.

For those forms, `domain_url` and `privacy_policy_url` may be supplied beside the snapshot or overridden on the command line.

## Local inference

```bash
python infer.py \
  --adapter models/veilance-qwen3-1.7b \
  --input examples/inference_input.json
```

The command writes only the final JSON report to stdout. Retrieval progress goes to stderr.

Useful options:

- `--domain-url` and `--privacy-policy-url` override input values.
- `--browser chromium|firefox|webkit` selects the installed Playwright browser.
- `--no-search` disables only the search fallback; supplied URL, first-party links, and common paths are still tried.
- `--retrieved-policy-output path.json` saves the exact host-prepared model input for debugging.
- `--allow-private-network` permits private or localhost targets for controlled development only.
- `--use-supplied-policy-document` skips Playwright and is intended only for offline tests and prepared fixtures.

The normal retrieval order is:

1. supplied `privacy_policy_url`;
2. privacy/legal links found on the domain homepage;
3. `/privacy`, `/privacy-policy`, `/legal/privacy`, `/legal/privacy-policy`, and `/policies/privacy`;
4. privacy-policy candidates returned by the Playwright web-search fallback.

Every candidate must verify as privacy-policy content and appear applicable to the supplied service. Search snippets are never used as policy evidence. An invalid, unsafe, stale, unrelated, or inaccessible supplied URL does not stop discovery.

### Python integration

Load the adapter once and reuse the engine for requests:

```python
from infer import PrivacyPolicyComparisonEngine

engine = PrivacyPolicyComparisonEngine(
    adapter="models/veilance-qwen3-1.7b"
)

privacy_results = engine.privacy_policy_comparison(res)
```

Here `res` is the same dictionary returned by the existing `normalize_telemetry_data(...)` function.

For custom retrieval settings:

```python
from infer import PrivacyPolicyComparisonEngine
from policy_retrieval import PolicyRetriever

retriever = PolicyRetriever(
    timeout_ms=20000,
    max_policy_chars=24000,
    search_enabled=True,
    browser_name="chromium"
)
engine = PrivacyPolicyComparisonEngine(
    adapter="models/veilance-qwen3-1.7b",
    retriever=retriever
)
privacy_results = engine.privacy_policy_comparison(res)
```

## Returned data

The result has exactly these top-level keys:

- `domain`
- `privacy_policy`
- `visit`
- `analysis`
- `findings`
- `important_limitations`

`schema.py` enforces the allowed policy, telemetry, comparison, and severity values; exact key sets; raw URL formatting; finding/count consistency; and valid disclosure relationships. `infer.py` also overwrites host-authoritative domain, policy, and visit fields, recomputes counts and overall confidence, checks cited section names against Playwright-extracted headings, and appends retrieval limitations.

If no applicable policy can be found, the output uses:

- `privacy_policy.found: false`
- `privacy_policy.applicable: false`
- `policy.status: "unknown"`
- `comparison: "indeterminate"`

It does not fabricate policy language.

## Raw training rows

Each source JSONL row contains:

```json
{
  "telemetry": {
    "schemaVersion": "veilance.telemetry-snapshot.v2"
  },
  "policy_document": {
    "url": "https://example.com/privacy",
    "found": true,
    "applicable": true,
    "sections": [
      {
        "heading": "Information We Collect",
        "text": "Policy text extracted by the host."
      }
    ]
  },
  "expected": {
    "domain": "https://example.com"
  }
}
```

The full telemetry object must use the real v2 names, including `resourceTypes` and `serviceWorkerControlled`. Synthetic-only aliases such as `types` and `serviceWorker` are rejected.

`prepare_dataset.py` projects each exact snapshot into the connector-shaped runtime input before creating the SFT prompt. Therefore training and inference use the same field layout. It validates policy grounding, the output report, maximum policy size, visit duration, and policy metadata before writing any split.

## Generate coherent synthetic rows

The generator accepts:

- a Veilance report object;
- a database row containing `policy_raw_results`;
- an existing training row containing `expected`.

```bash
python generate_synthetic_dataset.py seed_reports/ \
  --output data/raw.jsonl \
  --config generator_config.json \
  --seed 1337 \
  --fail-on-invalid
```

Each scenario generates telemetry, applicable policy language, and its expected finding together. That keeps API counts, network hosts, tracker categories, disclosure state, telemetry state, comparison, evidence, and limitations mutually consistent. Host and tracker records describing the same requests are counted once in `observation_count`.

The old misspelled `generate_sythetic_dataset.py` filename remains as a compatibility wrapper.

Use at least three independent source reports or domains. Synthetic descendants retain a `family_id`, and all descendants from one source stay in the same split to prevent train/test leakage.

## Prepare, train, and evaluate

```bash
python prepare_dataset.py \
  --input data/raw.jsonl \
  --output-dir data/processed

python train.py \
  --train data/processed/train.jsonl \
  --eval data/processed/validation.jsonl \
  --output models/veilance-qwen3-1.7b \
  --epochs 3 \
  --max-length 8192 \
  --learning-rate 1e-4 \
  --lora-r 32

python evaluate.py \
  --adapter models/veilance-qwen3-1.7b \
  --test data/processed/test.jsonl
```

Training disables Qwen thinking output and computes loss on the assistant completion only. Overlength samples fail validation instead of silently truncating JSON.

The evaluator reports schema validity, comparison accuracy, macro F1, policy/telemetry status accuracy, missing and extra findings, and false-accusation rate.

## Extract a real upload

```bash
python extract_telemetry.py telemetry.bin --output observations.jsonl
```

The extractor accepts gzip `telemetry.bin`, uncompressed batch JSON, and a single snapshot JSON without renaming telemetry fields.

## Safety and interpretation

The inference host rejects credential-bearing URLs, non-HTTP protocols, nonstandard ports, and public targets that resolve to private, loopback, link-local, or otherwise non-global addresses. Redirects and browser subrequests are checked as well. Keep `--allow-private-network` disabled in production.

The analysis remains conservative:

- observed activity is not proof of purpose;
- third-party contact is not proof that personal data was transmitted;
- API access is not automatically tracking or fingerprinting;
- storage state may predate the visit;
- one short visit cannot prove a behavior never occurs;
- the interest score is metadata, not proof of a privacy-policy discrepancy;
- the report makes no automatic legal or compliance conclusion.
