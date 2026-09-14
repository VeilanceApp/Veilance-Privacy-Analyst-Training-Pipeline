import argparse, json
from telemetry import load_telemetry_file

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--output", required=True)
args = ap.parse_args()
obs = load_telemetry_file(args.input)

with open(args.output, "w", encoding="utf-8") as f:
    for item in obs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Wrote {len(obs)} observation(s) to {args.output}")
