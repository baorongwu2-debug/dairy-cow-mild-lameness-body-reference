"""Cow-grouped, nested evaluation of body-referenced pose descriptors."""
from run_study import ROOT, digest, save_json, write_csv
import csv, json, time, collections, platform
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_curve
from threadpoolctl import threadpool_limits
import joblib, sklearn

DATA=ROOT/'external_data'/'official'/'lstm-lameness-detection-main'/'data'
OUT=ROOT/'pose_results'
KP=['LFHoof','RFHoof','LHHoof','RHHoof','Nose','HeadTop','Spine1','Spine2','Spine3']
SEEDS=[42,123,456,789,2024]

def load_data():
    with open(DATA/'videos_lameness_scores.csv',encoding='utf-8-sig') as f: labels=list(csv.DictReader(f))
    trajectories=[]; frames=[]; manifest=[]
    for r in labels:
        p=DATA/'videos_keypoints'/f"{r['Video']}.csv"
        with open(p,encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        x=np.array([[[float(v[k+'_x']),float(v[k+'_y'])] for k in KP] for v in rows])
        t=np.array([float(v['frame']) for v in rows])/30.
        if not np.isfinite(x).all() or len(t)<31 or not (np.diff(t)>0).all(): raise ValueError(f'Invalid trajectory {p}')
        trajectories.append(x); frames.append(t)
        manifest.append(dict(video=r['Video'],cow_id=r['ID'],score=int(r['hard_vote']),frames=len(t),duration=float(t[-1]-t[0]),source_video=rows[0]['video'],sha256=digest(p),frame_gaps=int(np.sum(np.diff(np.array([float(v['frame']) for v in rows]))!=1))))
    assert len(set(r['video'] for r in manifest))==len(manifest)
    assert len(set(r['sha256'] for r in manifest))==len(manifest)
    return trajectories,frames,manifest

def coordinates(x,canonical=True):
    x=savgol_filter(x,7,2,axis=0,mode='interp')
    if not canonical: return x.reshape(len(x),-1)
    trunk=x[:,6]-x[:,7]; lengths=np.linalg.norm(trunk,axis=1)
    if lengths.min()<1e-6: raise ValueError('Degenerate trunk coordinates')
    origin=(x[:,6]+x[:,7])/2; ex=trunk/lengths[:,None]; ey=np.stack([-ex[:,1],ex[:,0]],axis=1)
    relative=x-origin[:,None,:]
    transformed=np.stack([np.einsum('tkd,td->tk',relative,ex),np.einsum('tkd,td->tk',relative,ey)],axis=2)/np.median(lengths)
    return transformed.reshape(len(x),-1)

def stats(x):
    return np.concatenate([np.median(x,axis=0),np.percentile(x,75,axis=0)-np.percentile(x,25,axis=0),np.percentile(x,10,axis=0),np.percentile(x,90,axis=0)])

def describe(x,t,canonical=True):
    c=coordinates(x,canonical); velocity=np.gradient(c,t,axis=0); acceleration=np.gradient(velocity,t,axis=0)
    corrs=[]
    for lag in [1,5,15,30]:
        a=c[:-lag]-c[:-lag].mean(0); b=c[lag:]-c[lag:].mean(0)
        denom=np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0))
        corrs.append(np.divide(np.sum(a*b,axis=0),denom,out=np.zeros(c.shape[1]),where=denom>1e-10))
    static=stats(c)
    dynamic=np.concatenate([stats(velocity),np.percentile(acceleration,75,axis=0)-np.percentile(acceleration,25,axis=0),*corrs])
    return static,dynamic

def measures(y,s):
    pred=s>=0; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(auc=float(roc_auc_score(y,s)),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)))

def classifier(kind):
    if kind=='lr': return Pipeline([('variance',VarianceThreshold(1e-12)),('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]),{'model__C':[.01,.1,1,10]}
    if kind=='rf': return Pipeline([('variance',VarianceThreshold(1e-12)),('scale',StandardScaler()),('model',RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=1))]),{'model__max_depth':[3,None],'model__min_samples_leaf':[2,5]}
    return Pipeline([('variance',VarianceThreshold(1e-12)),('scale',StandardScaler()),('model',SVC(class_weight='balanced'))]),{'model__C':[.1,1,10,100],'model__gamma':['scale',.01,.1]}

def score(clf,X):
    if hasattr(clf,'decision_function'): return clf.decision_function(X)
    return clf.predict_proba(X)[:,1]-.5

def main():
    start=time.time(); OUT.mkdir(exist_ok=True)
    xs,ts,manifest=load_data(); write_csv(OUT/'manifest.csv',manifest)
    static=[]; dynamic=[]; raw=[]; perturb=[]
    theta=np.deg2rad(20); rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    for x,t in zip(xs,ts):
        a,b=describe(x,t); static.append(a); dynamic.append(b)
        raw.append(np.concatenate(describe(x,t,False)))
        perturb.append(np.concatenate(describe(1.5*x@rotation.T+np.array([250,-150]),t)))
    static=np.array(static); dynamic=np.array(dynamic); combined=np.concatenate([static,dynamic],axis=1); raw=np.array(raw); perturb=np.array(perturb)
    # Constant-coordinate correlations can amplify floating point noise; test relative, not pixel-scale, descriptors.
    np.testing.assert_allclose(combined,perturb,atol=1e-6,rtol=1e-6)
    np.savez_compressed(OUT/'features.npz',static=static,dynamic=dynamic,combined=combined,raw=raw)
    labels=np.array([r['score'] for r in manifest]); groups=np.array([r['cow_id'] for r in manifest]); names=np.array([r['video'] for r in manifest])
    save_json(OUT/'data_audit.json',dict(n=len(labels),cows=len(set(groups)),scores=dict(collections.Counter(map(str,labels))),mild_n=int(sum(labels<=2)),mild_cows=len(set(groups[labels<=2])),frame_gaps=sum(r['frame_gaps'] for r in manifest),duration_range=[min(r['duration'] for r in manifest),max(r['duration'] for r in manifest)],canonical_transform_max_error=float(np.max(np.abs(combined-perturb))),archive_sha256=digest(ROOT/'external_data'/'russello_official.zip') if (ROOT/'external_data'/'russello_official.zip').exists() else None,label_sha256=digest(DATA/'videos_lameness_scores.csv'),protocol_sha256=digest(ROOT/'pose_protocol.md')))
    configs={'body_combined_svm':(combined,'svm'),'image_combined_svm':(raw,'svm'),'body_static_svm':(static,'svm'),'body_dynamic_svm':(dynamic,'svm'),'body_combined_lr':(combined,'lr'),'body_combined_rf':(combined,'rf')}
    summaries={}; predictions=[]; split_records=[]; primary_models=[]
    for endpoint in ['mild','all']:
        ix=np.flatnonzero(labels<=2) if endpoint=='mild' else np.arange(len(labels)); y=(labels[ix]>1).astype(int); g=groups[ix]
        endpoint_scores={}; summaries[endpoint]={}
        for name,(fullX,kind) in configs.items():
            X=fullX[ix]; estimator,grid=classifier(kind); endpoint_scores[name]=[]
            for seed in SEEDS:
                s=np.full(len(y),np.nan)
                outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,g))
                for fold,(tr,te) in enumerate(outer):
                    assert not set(g[tr])&set(g[te])
                    inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed+fold).split(X[tr],y[tr],g[tr]))
                    for it,iv in inner:
                        assert not set(g[tr][it])&set(g[tr][iv])
                        assert len(set(y[tr][it]))==len(set(y[tr][iv]))==2
                    search=GridSearchCV(estimator,grid,cv=inner,scoring='roc_auc',n_jobs=1,error_score='raise'); search.fit(X[tr],y[tr]); s[te]=score(search,X[te])
                    split_records.append(dict(endpoint=endpoint,model=name,seed=seed,fold=fold,train=ix[tr].tolist(),test=ix[te].tolist(),params=search.best_params_,inner=[dict(train=ix[tr[it]].tolist(),validation=ix[tr[iv]].tolist()) for it,iv in inner]))
                    predictions.extend(dict(endpoint=endpoint,model=name,seed=seed,fold=fold,video=str(names[ix[j]]),cow_id=str(g[j]),label=int(y[j]),score=float(s[j])) for j in te)
                    if seed==42 and name=='body_combined_svm':
                        p=score(search,perturb[ix[te]]); np.testing.assert_allclose(s[te],p,atol=1e-6,rtol=1e-6)
                        joblib.dump(search.best_estimator_,OUT/f'{endpoint}_seed42_fold{fold}.joblib')
                assert np.isfinite(s).all(); endpoint_scores[name].append(s)
                print(endpoint,name,seed,measures(y,s),flush=True)
                write_csv(OUT/'predictions.csv',predictions); save_json(OUT/'splits.json',split_records)
            ms=[measures(y,s) for s in endpoint_scores[name]]
            summaries[endpoint][name]={k:dict(mean=float(np.mean([v[k] for v in ms])),split_sd=float(np.std([v[k] for v in ms],ddof=1))) for k in ms[0]}
            summaries[endpoint][name]['seed42']=ms[0]
        # Resample whole cow clusters; repeated selections retain multiplicity.
        rng=np.random.default_rng(20260907); unique=np.unique(g); cluster={u:np.flatnonzero(g==u) for u in unique}
        boot={name:[] for name in configs}; deltas=[]
        for _ in range(2000):
            b=np.concatenate([cluster[u] for u in rng.choice(unique,len(unique),replace=True)])
            if len(set(y[b]))<2: continue
            for name in configs: boot[name].append(roc_auc_score(y[b],endpoint_scores[name][0][b]))
            deltas.append(boot['body_combined_svm'][-1]-boot['body_static_svm'][-1])
        for name in configs: summaries[endpoint][name]['cow_bootstrap_auc_interval']=np.quantile(boot[name],[.025,.975]).tolist()
        summaries[endpoint]['combined_minus_static']=dict(auc_difference=float(roc_auc_score(y,endpoint_scores['body_combined_svm'][0])-roc_auc_score(y,endpoint_scores['body_static_svm'][0])),cow_bootstrap_interval=np.quantile(deltas,[.025,.975]).tolist())
        save_json(OUT/'summary.json',summaries)
    save_json(OUT/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,script_sha256=digest(Path(__file__)),elapsed_seconds=time.time()-start))
    print('Completed all cow-grouped experiments.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
