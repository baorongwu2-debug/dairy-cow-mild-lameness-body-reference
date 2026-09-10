"""Fixed compact BiLSTM and temporal CNN baselines under cow-grouped folds."""
from run_study import ROOT, save_json, write_csv, digest
from pose_study import load_data, coordinates, measures
import time, json, random, platform
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT=ROOT/'sequence_results'; SEED=42; LENGTH=90
class BiLSTM(nn.Module):
    def __init__(self,d):
        super().__init__(); self.rnn=nn.LSTM(d,32,1,batch_first=True,bidirectional=True); self.out=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.25),nn.Linear(32,1))
    def forward(self,x): return self.out(self.rnn(x)[0].mean(1)).squeeze(1)
class TemporalCNN(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(d,32,5,padding=2),nn.ReLU(),nn.BatchNorm1d(32),nn.Conv1d(32,32,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.out=nn.Linear(32,1)
    def forward(self,x): return self.out(self.net(x.transpose(1,2)).squeeze(2)).squeeze(1)
def resample(x):
    c=coordinates(x); old=np.linspace(0,1,len(c)); new=np.linspace(0,1,LENGTH)
    return np.stack([np.interp(new,old,c[:,j]) for j in range(c.shape[1])],1)
def seed_all(v): random.seed(v); np.random.seed(v); torch.manual_seed(v)
def fit_predict(cls,X,y,g,tr,te,fold):
    seed_all(SEED+fold); inner=next(StratifiedGroupKFold(5,shuffle=True,random_state=900+fold).split(X[tr],y[tr],g[tr])); fit=tr[inner[0]]; va=tr[inner[1]]
    mean=X[fit].reshape(-1,X.shape[-1]).mean(0); sd=X[fit].reshape(-1,X.shape[-1]).std(0); sd[sd<1e-8]=1
    Z=(X-mean)/sd; xt=torch.tensor(Z,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32)
    model=cls(X.shape[-1]); pos=float((y[fit]==0).sum()/max(1,(y[fit]==1).sum())); loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=None; patience=0
    for epoch in range(80):
        model.train(); order=np.random.permutation(fit)
        for start in range(0,len(order),16):
            b=order[start:start+16]; opt.zero_grad(); v=loss(model(xt[b]),yt[b]); v.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(loss(model(xt[va]),yt[va]))
        if best is None or vl<best[0]-1e-5: best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch); patience=0
        else: patience+=1
        if patience>=12: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): score=model(xt[te]).numpy()
    return score,best[2],len(fit),len(va)
def main():
    start=time.time(); OUT.mkdir(exist_ok=True); xs,ts,manifest=load_data(); X=np.array([resample(x) for x in xs]); labels=np.array([r['score'] for r in manifest]); g=np.array([r['cow_id'] for r in manifest]); videos=np.array([r['video'] for r in manifest]); rows=[]; summary={}; splits=[]
    for endpoint in ['mild','all']:
        ix=np.flatnonzero(labels<=2) if endpoint=='mild' else np.arange(len(labels)); y=(labels[ix]>1).astype(int); scores={}
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
        for name,cls in [('compact_bilstm',BiLSTM),('temporal_cnn',TemporalCNN)]:
            oof=np.zeros(len(ix))
            for fold,(tr,te) in enumerate(outer.split(X[ix],y,g[ix])):
                sc,epoch,nfit,nval=fit_predict(cls,X[ix],y,g[ix],tr,te,fold); oof[te]=sc
                splits.append(dict(endpoint=endpoint,model=name,seed=SEED,fold=fold,train=ix[tr].tolist(),test=ix[te].tolist(),best_epoch=epoch,fit_n=nfit,validation_n=nval))
                for j,s in zip(te,sc): rows.append(dict(endpoint=endpoint,model=name,seed=SEED,fold=fold,video=videos[ix[j]],cow_id=g[ix[j]],label=int(y[j]),score=float(s)))
            summary.setdefault(endpoint,{})[name]=measures(y,oof)
    write_csv(OUT/'predictions.csv',rows); save_json(OUT/'splits.json',splits); save_json(OUT/'summary.json',summary); save_json(OUT/'environment.json',dict(runtime_seconds=time.time()-start,torch=torch.__version__,seed=SEED,sequence_length=LENGTH,protocol_sha256=digest(ROOT/'enhanced_protocol.md'))); print(json.dumps(summary,indent=2))
if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main()
