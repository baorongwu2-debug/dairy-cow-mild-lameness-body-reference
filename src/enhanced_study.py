"""Block-aware fusion, strong classical baselines, and same-farm session holdout."""
from run_study import ROOT, save_json, write_csv, digest
from pose_study import load_data, measures, SEEDS
import json, csv, time, collections
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import clone
from threadpoolctl import threadpool_limits

OUT=ROOT/'enhanced_results'

def fit_blocks(a,b):
    vp=VarianceThreshold(1e-12).fit(a); vm=VarianceThreshold(1e-12).fit(b)
    sp=StandardScaler().fit(vp.transform(a)); sm=StandardScaler().fit(vm.transform(b))
    return (vp,sp),(vm,sm)
def transform(pair,x): return pair[1].transform(pair[0].transform(x))
def bandwidth(x):
    d=euclidean_distances(x,squared=True); v=d[np.triu_indices_from(d,1)]; v=v[v>1e-12]
    return float(np.median(v)) if len(v) else 1.
def kernel(a,b,den,mult): return np.exp(-mult*euclidean_distances(a,b,squared=True)/den)
def fit_mkl(p,m,y,C,alpha,mult):
    pb,mb=fit_blocks(p,m); ps=transform(pb,p); ms=transform(mb,m); bp=bandwidth(ps); bm=bandwidth(ms)
    K=alpha*kernel(ps,ps,bp,mult)+(1-alpha)*kernel(ms,ms,bm,mult)
    model=SVC(kernel='precomputed',C=C,class_weight='balanced').fit(K,y)
    return dict(pb=pb,mb=mb,ps=ps,ms=ms,bp=bp,bm=bm,model=model,alpha=alpha,mult=mult,C=C)
def score_mkl(fit,p,m):
    ps=transform(fit['pb'],p); ms=transform(fit['mb'],m)
    K=fit['alpha']*kernel(ps,fit['ps'],fit['bp'],fit['mult'])+(1-fit['alpha'])*kernel(ms,fit['ms'],fit['bm'],fit['mult'])
    return fit['model'].decision_function(K)
def tune_mkl(p,m,y,g,seed):
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(p,y,g)); best=None
    for alpha in [.75,.9,1.]:
      for mult in [1.]:
       for C in [.1,1.,10.]:
        scores=[]
        for tr,va in folds:
            f=fit_mkl(p[tr],m[tr],y[tr],C,alpha,mult); scores.append(roc_auc_score(y[va],score_mkl(f,p[va],m[va])))
        item=(float(np.mean(scores)),alpha,mult,C)
        if best is None or item>best: best=item
    return {'alpha':best[1],'mult':best[2],'C':best[3],'inner_auc':best[0]}
def classical():
    common=[('variance',VarianceThreshold(1e-12)),('scale',StandardScaler())]
    return {
      'posture_svm':(Pipeline(common+[('model',SVC(class_weight='balanced'))]),{'model__C':[.1,1,10,100],'model__gamma':['scale',.01,.1]}),
      'posture_extratrees':(Pipeline([('variance',VarianceThreshold(1e-12)),('model',ExtraTreesClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=1))]),{'model__max_features':['sqrt',.5],'model__min_samples_leaf':[1,5]}),
      'posture_knn':(Pipeline(common+[('model',KNeighborsClassifier())]),{'model__n_neighbors':[3,5,9,15],'model__weights':['uniform','distance']}),
      'posture_gnb':(Pipeline(common+[('model',GaussianNB())]),{'model__var_smoothing':[1e-11,1e-9,1e-7,1e-5]})}
def cscore(model,x):
    return model.decision_function(x) if hasattr(model,'decision_function') else model.predict_proba(x)[:,1]-.5

def main():
    start=time.time(); OUT.mkdir(exist_ok=True); xs,ts,manifest=load_data(); z=np.load(ROOT/'pose_results'/'features.npz')
    labels=np.array([r['score'] for r in manifest]); cows=np.array([r['cow_id'] for r in manifest]); videos=np.array([r['video'] for r in manifest]); sessions=np.array([r['source_video'][:4] for r in manifest])
    predictions=[]; splits=[]; summary={}
    for endpoint in ['mild']:
      ix=np.flatnonzero(labels<=2) if endpoint=='mild' else np.arange(len(labels)); y=(labels[ix]>1).astype(int); g=cows[ix]; p=z['static'][ix]; m=z['dynamic'][ix]
      summary[endpoint]={}; store=collections.defaultdict(list)
      for seed in SEEDS:
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        for fold,(tr,te) in enumerate(outer.split(p,y,g)):
          prm=tune_mkl(p[tr],m[tr],y[tr],g[tr],seed+fold); f=fit_mkl(p[tr],m[tr],y[tr],prm['C'],prm['alpha'],prm['mult']); sc=score_mkl(f,p[te],m[te]); store[('mkl',seed)].extend(zip(te,sc)); splits.append(dict(endpoint=endpoint,model='mkl',seed=seed,fold=fold,train=ix[tr].tolist(),test=ix[te].tolist(),params=prm))
          for j,s in zip(te,sc): predictions.append(dict(endpoint=endpoint,validation='cow_grouped',model='mkl',seed=seed,fold=fold,video=videos[ix[j]],cow_id=g[j],label=int(y[j]),score=float(s)))
          inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed+fold).split(p[tr],y[tr],g[tr]))
          for name,(est,grid) in classical().items():
            if seed != 42 and name not in ['posture_svm']: continue
            q=GridSearchCV(est,grid,cv=inner,scoring='roc_auc',error_score='raise',n_jobs=1).fit(p[tr],y[tr]); sc=cscore(q.best_estimator_,p[te]); store[(name,seed)].extend(zip(te,sc)); splits.append(dict(endpoint=endpoint,model=name,seed=seed,fold=fold,train=ix[tr].tolist(),test=ix[te].tolist(),params=q.best_params_))
            for j,s in zip(te,sc): predictions.append(dict(endpoint=endpoint,validation='cow_grouped',model=name,seed=seed,fold=fold,video=videos[ix[j]],cow_id=g[j],label=int(y[j]),score=float(s)))
      for name in ['mkl',*classical().keys()]:
        vals=[]
        model_seeds=SEEDS if name in ['mkl','posture_svm'] else [42]
        for seed in model_seeds:
            pairs=sorted(store[(name,seed)]); assert [j for j,_ in pairs]==list(range(len(ix))); vals.append(measures(y,np.array([s for _,s in pairs])))
        summary[endpoint][name]={k:{'mean':float(np.mean([v[k] for v in vals])),'split_sd':float(np.std([v[k] for v in vals],ddof=1))} for k in vals[0]}
      # Strict acquisition-session holdout, same farm. Proposed and comparator only.
      session_scores={name:np.full(len(ix),np.nan) for name in ['mkl','posture_svm']}; audit=[]
      for fold,session in enumerate(sorted(set(sessions[ix]))):
        te=np.flatnonzero(sessions[ix]==session); test_cows=set(g[te]); tr=np.array([j for j in range(len(ix)) if sessions[ix[j]]!=session and g[j] not in test_cows])
        if len(set(y[tr]))<2 or len(set(y[te]))<2: audit.append(dict(session=session,status='skipped_single_class',train=len(tr),test=len(te))); continue
        prm=tune_mkl(p[tr],m[tr],y[tr],g[tr],100+fold); f=fit_mkl(p[tr],m[tr],y[tr],prm['C'],prm['alpha'],prm['mult']); session_scores['mkl'][te]=score_mkl(f,p[te],m[te])
        inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=100+fold).split(p[tr],y[tr],g[tr])); est,grid=classical()['posture_svm']; q=GridSearchCV(est,grid,cv=inner,scoring='roc_auc').fit(p[tr],y[tr]); session_scores['posture_svm'][te]=cscore(q.best_estimator_,p[te])
        audit.append(dict(session=session,status='evaluated',train=len(tr),test=len(te),train_cows=len(set(g[tr])),test_cows=len(test_cows),cow_overlap=len(set(g[tr])&test_cows),mkl_params=prm,posture_params=q.best_params_))
        for name in session_scores:
          for j in te: predictions.append(dict(endpoint=endpoint,validation='session_holdout',model=name,seed=100,fold=fold,video=videos[ix[j]],cow_id=g[j],label=int(y[j]),score=float(session_scores[name][j])))
      valid=np.isfinite(session_scores['mkl']); summary[endpoint]['session_holdout']={'n':int(valid.sum()),'sessions':audit,'mkl':measures(y[valid],session_scores['mkl'][valid]),'posture_svm':measures(y[valid],session_scores['posture_svm'][valid])}
    write_csv(OUT/'predictions.csv',predictions); save_json(OUT/'splits.json',splits); save_json(OUT/'summary.json',summary); save_json(OUT/'environment.json',dict(runtime_seconds=time.time()-start,protocol_sha256=digest(ROOT/'enhanced_protocol.md'),feature_sha256=digest(ROOT/'pose_results'/'features.npz')))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
  with threadpool_limits(limits=1): main()
