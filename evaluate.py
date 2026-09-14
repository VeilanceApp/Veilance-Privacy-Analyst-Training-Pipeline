import argparse,json
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
from infer import load_model,analyze


def rows(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def expected(r):
    return json.loads(r['completion'][-1]['content'])


def inp(r):
    text=next(m['content'] for m in r['prompt'] if m['role']=='user'); marker='INPUT:\n'; return json.loads(text[text.index(marker)+len(marker):])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-model',default='Qwen/Qwen3-1.7B');ap.add_argument('--adapter',required=True);ap.add_argument('--test',required=True);args=ap.parse_args()
    tok,model=load_model(args.base_model,args.adapter); yt=[];yp=[];invalid=missing=extra=false_acc=accusations=0
    for r in tqdm(list(rows(args.test))):
        gold=expected(r)
        try: pred=analyze(tok,model,inp(r))
        except Exception: invalid+=1;continue
        g={x['behavior']:x for x in gold['findings']};p={x['behavior']:x for x in pred['findings']};missing+=len(set(g)-set(p));extra+=len(set(p)-set(g))
        for b,x in g.items():
            if b not in p:continue
            yt.append(x['comparison']);yp.append(p[b]['comparison'])
            if p[b]['comparison'] in {'observed_only','possible_contradiction'}:
                accusations+=1
                if x['comparison'] not in {'observed_only','possible_contradiction'}:false_acc+=1
    print(json.dumps({'invalid_outputs':invalid,'matched_findings':len(yt),'missing_findings':missing,'extra_findings':extra,'comparison_accuracy':accuracy_score(yt,yp) if yt else 0,'comparison_macro_f1':f1_score(yt,yp,average='macro',zero_division=0) if yt else 0,'false_accusations':false_acc,'predicted_accusations':accusations,'false_accusation_rate':false_acc/accusations if accusations else 0},indent=2))


if __name__=='__main__':
    main()
