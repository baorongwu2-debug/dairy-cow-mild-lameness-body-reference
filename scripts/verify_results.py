"""Independent saved-result checks using only the Python standard library."""
from pathlib import Path
import csv, json, math, collections, statistics, ast
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def auc(rows):
    ordered=sorted(rows,key=lambda r:float(r['score'])); rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and float(ordered[j]['score'])==float(ordered[i]['score']): j+=1
        rank_sum+=(i+1+j)/2*sum(int(r['label']) for r in ordered[i:j]);i=j
    pos=sum(int(r['label']) for r in rows);neg=len(rows)-pos
    assert pos and neg
    return (rank_sum-pos*(pos+1)/2)/(pos*neg)
def main():
    base=ROOT/'results'; manifest=read(base/'pose_results/manifest.csv'); mild=[r for r in manifest if int(r['score'])<=2]
    assert len(manifest)==272 and len(mild)==239
    assert len({r['cow_id'] for r in manifest})==98 and len({r['cow_id'] for r in mild})==92
    lookup={r['video']:r for r in manifest};count=0;folds=0;aucs={};report={}
    for folder in ['pose_results','enhanced_results','sequence_results','fusion_results']:
        rows=read(base/folder/'predictions.csv'); count+=len(rows)
        assert len(rows)=={'pose_results':15330,'enhanced_results':3585,'sequence_results':1022,'fusion_results':239}[folder]
        groups=collections.defaultdict(list)
        for r in rows:
            original=lookup[r['video']]
            assert r['cow_id']==original['cow_id'] and int(r['label'])==int(int(original['score'])>1)
            assert math.isfinite(float(r['score']))
            groups[(r['endpoint'],r.get('validation','cow_grouped'),r['model'],r['seed'])].append(r)
        sums=json.loads((base/folder/'summary.json').read_text())
        average=collections.defaultdict(list)
        for (ep,validation,model,seed),rr in groups.items():
            expected=mild if ep=='mild' else manifest
            assert len(rr)==len(expected) and {r['video'] for r in rr}=={r['video'] for r in expected}
            assert len({r['video'] for r in rr})==len(rr)
            score=auc(rr); average[(ep,validation,model)].append(score)
            aucs[f'{folder}/{ep}/{validation}/{model}/{seed}']=score
        for (ep,validation,model),values in average.items():
            if folder=='fusion_results': expected=sums['auc']
            elif folder=='sequence_results': expected=sums[ep][model]['auc']
            elif validation=='session_holdout': expected=sums[ep]['session_holdout'][model]['auc']
            else: expected=sums[ep][model]['auc']['mean']
            assert abs(statistics.mean(values)-expected)<1e-12,(folder,model,values,expected)
        split=json.loads((base/folder/'splits.json').read_text())
        index=mild if folder=='fusion_results' else manifest
        for s in split:
            tr=s['train'];te=s['test'];assert not set(tr)&set(te)
            assert not {index[j]['cow_id'] for j in tr}&{index[j]['cow_id'] for j in te}
            for inner in s.get('inner',[]):
                a=inner['train'];b=inner['validation'];assert set(a)|set(b)==set(tr)
                assert not {index[j]['cow_id'] for j in a}&{index[j]['cow_id'] for j in b}
            # Tie each split to its prediction identities, not just stored overlap counters.
            ep=s.get('endpoint','mild');model=s.get('model','validation_gated_posture_cnn');seed=str(s.get('seed',42))
            rr=[r for r in groups[(ep,'cow_grouped',model,seed)] if int(r['fold'])==s['fold']]
            assert {r['video'] for r in rr}=={index[j]['video'] for j in te}
            folds+=1
        if folder=='enhanced_results':
            sessions=sums['mild']['session_holdout']['sessions']
            for k,s in enumerate(sessions):
                assert s['status']=='evaluated'
                test=[r for r in mild if r['source_video'][:4]==s['session']]; ids={r['cow_id'] for r in test}
                train=[r for r in mild if r['source_video'][:4]!=s['session'] and r['cow_id'] not in ids]
                assert len(test)==s['test'] and len(train)==s['train'] and not {r['cow_id'] for r in train}&ids
                for model in ['mkl','posture_svm']:
                    rr=[r for r in groups[('mild','session_holdout',model,'100')] if int(r['fold'])==k]
                    assert {r['video'] for r in rr}=={r['video'] for r in test}
        report[folder]={'predictions':len(rows),'auc_groups_checked':len(groups),'outer_splits_checked':len(split)}
    for p in ROOT.rglob('*.py'): ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    result={'status':'passed','scope':'Saved predictions and stored outer/inner splits; no model retraining in this check','prediction_rows':count,'outer_splits_checked':folds,'results':report,'recomputed_auc':aucs}
    (ROOT/'provenance/package_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='recomputed_auc'},indent=2))
if __name__=='__main__': main()
