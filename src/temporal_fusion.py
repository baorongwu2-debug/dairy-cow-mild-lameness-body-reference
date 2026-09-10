"""Exploratory validation-gated fusion of posture SVM and temporal CNN."""
from run_study import ROOT, save_json, write_csv, digest
from pose_study import load_data, coordinates, measures, classifier, score
from sequence_baselines import TemporalCNN, resample, seed_all, LENGTH
import numpy as np, torch, json, time
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

OUT=ROOT/'fusion_results'; SEED=42
def train_fold(X,P,y,g,tr,te,fold):
    seed_all(SEED+fold); a,b=next(StratifiedGroupKFold(5,shuffle=True,random_state=700+fold).split(X[tr],y[tr],g[tr])); fit=tr[a]; va=tr[b]
    mean=X[fit].reshape(-1,X.shape[-1]).mean(0); sd=X[fit].reshape(-1,X.shape[-1]).std(0); sd[sd<1e-8]=1
    xt=torch.tensor((X-mean)/sd,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32); model=TemporalCNN(X.shape[-1]); pos=float((y[fit]==0).sum()/max(1,(y[fit]==1).sum())); loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=None; patience=0
    for epoch in range(80):
        model.train(); order=np.random.permutation(fit)
        for st in range(0,len(order),16):
            q=order[st:st+16]; opt.zero_grad(); v=loss(model(xt[q]),yt[q]); v.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(loss(model(xt[va]),yt[va]))
        if best is None or vl<best[0]-1e-5: best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch); patience=0
        else: patience+=1
        if patience>=12: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): cnn_va=model(xt[va]).numpy(); cnn_te=model(xt[te]).numpy()
    svm,grid=classifier('svm'); inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=800+fold).split(P[fit],y[fit],g[fit])); q=GridSearchCV(svm,grid,cv=inner,scoring='roc_auc').fit(P[fit],y[fit]); svm_va=score(q.best_estimator_,P[va]); svm_te=score(q.best_estimator_,P[te])
    def norm(v,ref): return (v-ref.mean())/(ref.std()+1e-8)
    cv=norm(cnn_va,cnn_va); ct=norm(cnn_te,cnn_va); sv=norm(svm_va,svm_va); st=norm(svm_te,svm_va)
    choices=[]
    for alpha in [0,.25,.5,.75,1.]: choices.append((roc_auc_score(y[va],alpha*sv+(1-alpha)*cv),alpha))
    alpha=max(choices)[1]; return alpha*st+(1-alpha)*ct,dict(alpha=alpha,validation_auc=max(choices)[0],cnn_epoch=best[2],svm_params=q.best_params_,fit_n=len(fit),gate_validation_n=len(va))
def main():
    start=time.time(); OUT.mkdir(exist_ok=True); xs,ts,manifest=load_data(); labels=np.array([r['score'] for r in manifest]); ix=np.flatnonzero(labels<=2); y=(labels[ix]>1).astype(int); g=np.array([r['cow_id'] for r in manifest])[ix]; videos=np.array([r['video'] for r in manifest])[ix]; X=np.array([resample(xs[j]) for j in ix]); P=np.load(ROOT/'pose_results'/'features.npz')['static'][ix]; oof=np.zeros(len(ix)); rows=[]; splits=[]
    for fold,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,g)):
        sc,params=train_fold(X,P,y,g,tr,te,fold); oof[te]=sc; splits.append(dict(fold=fold,train=tr.tolist(),test=te.tolist(),params=params))
        for j,s in zip(te,sc): rows.append(dict(endpoint='mild',model='validation_gated_posture_cnn',seed=SEED,fold=fold,video=videos[j],cow_id=g[j],label=int(y[j]),score=float(s)))
    result=measures(y,oof); result['gate_alphas']=[s['params']['alpha'] for s in splits]; write_csv(OUT/'predictions.csv',rows); save_json(OUT/'splits.json',splits); save_json(OUT/'summary.json',result); save_json(OUT/'environment.json',dict(runtime_seconds=time.time()-start,protocol_sha256=digest(ROOT/'enhanced_protocol.md'),status='exploratory development after earlier results; requires independent confirmation')); print(json.dumps(result,indent=2))
if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main()
