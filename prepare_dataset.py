import argparse, json, random
from collections import defaultdict
from pathlib import Path
from prompt import SYSTEM_PROMPT, build_user_prompt, compact_json
from schema import validate_report
from telemetry import validate_snapshot_shape


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): continue
            try: yield json.loads(line)
            except Exception as e: raise ValueError(f"{path}:{n}: {e}") from e


def validate_row(row, i):
    for k in ["telemetry", "policy_document", "expected"]:
        if k not in row: raise ValueError(f"row {i}: missing {k}")
    validate_snapshot_shape(row["telemetry"])
    validate_report(row["expected"])
    host = row["telemetry"]["site"]["hostname"]
    if host not in row["expected"]["domain"]:
        raise ValueError(f"row {i}: expected.domain does not contain telemetry hostname {host}")


def to_sft(row):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)}
        ],
        "completion": [{"role": "assistant", "content": compact_json(row["expected"])}],
        "domain": row["telemetry"]["site"]["hostname"]
    }


def split(rows, seed, tr=0.9, vr=0.05):
    groups=defaultdict(list)
    for r in rows: groups[r["telemetry"]["site"]["hostname"].lower()].append(r)
    domains=list(groups); random.Random(seed).shuffle(domains)
    if len(domains)<3: raise ValueError("need at least 3 unique domains")
    nt=max(1,int(len(domains)*tr)); nv=max(1,int(len(domains)*vr))
    if nt+nv>=len(domains): nt=len(domains)-2; nv=1
    sets=[set(domains[:nt]),set(domains[nt:nt+nv]),set(domains[nt+nv:])]
    return [[r for d in s for r in groups[d]] for s in sets]


def write(path, rows):
    with open(path,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(to_sft(r),ensure_ascii=False)+"\n")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--seed",type=int,default=1337); args=ap.parse_args()
    rows=list(read_jsonl(args.input))
    for i,r in enumerate(rows): validate_row(r,i)
    train,val,test=split(rows,args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    write(out/'train.jsonl',train); write(out/'validation.jsonl',val); write(out/'test.jsonl',test)
    manifest={"rows":len(rows),"domains":len({r['telemetry']['site']['hostname'] for r in rows}),"train":len(train),"validation":len(val),"test":len(test)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
