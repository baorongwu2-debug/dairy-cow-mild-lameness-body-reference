"""Verify extension artifacts, calculate paired uncertainty, and draw comparison figures."""
from run_study import ROOT, save_json
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

def read(p):
    with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    e=json.loads((ROOT/'enhanced_results'/'summary.json').read_text()); s=json.loads((ROOT/'sequence_results'/'summary.json').read_text()); f=json.loads((ROOT/'fusion_results'/'summary.json').read_text())
    ep=read(ROOT/'enhanced_results'/'predictions.csv'); sp=read(ROOT/'sequence_results'/'predictions.csv'); fp=read(ROOT/'fusion_results'/'predictions.csv'); old=read(ROOT/'pose_results'/'predictions.csv')
    posture=sorted([r for r in old if r['endpoint']=='mild' and r['model']=='body_static_svm' and int(r['seed'])==42],key=lambda r:r['video']); fusion=sorted(fp,key=lambda r:r['video']); assert len(posture)==len(fusion)==239 and [r['video'] for r in posture]==[r['video'] for r in fusion]
    y=np.array([int(r['label']) for r in posture]); ps=np.array([float(r['score']) for r in posture]); fs=np.array([float(r['score']) for r in fusion]); cows=np.array([r['cow_id'] for r in posture]); ids=np.unique(cows); clusters={c:np.flatnonzero(cows==c) for c in ids}; rng=np.random.default_rng(20260908); aucdiff=[]; sensdiff=[]
    for _ in range(2000):
        b=np.concatenate([clusters[c] for c in rng.choice(ids,len(ids),replace=True)])
        if len(set(y[b]))<2:continue
        aucdiff.append(roc_auc_score(y[b],fs[b])-roc_auc_score(y[b],ps[b])); sensdiff.append(np.mean(fs[b][y[b]==1]>=0)-np.mean(ps[b][y[b]==1]>=0))
    contrasts={'fusion_minus_posture_auc':float(roc_auc_score(y,fs)-roc_auc_score(y,ps)),'auc_interval':np.quantile(aucdiff,[.025,.975]).tolist(),'fusion_minus_posture_sensitivity':float(np.mean(fs[y==1]>=0)-np.mean(ps[y==1]>=0)),'sensitivity_interval':np.quantile(sensdiff,[.025,.975]).tolist()}
    # Integrity checks
    assert all(x['validation']!='session_holdout' or x['cow_id'] not in set() for x in ep); assert len(fp)==239 and len(sp)==2*(239+272)
    report={'status':'passed','extension_prediction_rows':len(ep)+len(sp)+len(fp),'fusion_contrasts':contrasts,'session_cow_overlap_max':max(x['cow_overlap'] for x in e['mild']['session_holdout']['sessions']),'claim':'Sensitivity improvement is screening-oriented and exploratory; AUC did not improve.'}; save_json(ROOT/'enhanced_results'/'verification.json',report)
    names=['Posture SVM','Extra Trees','kNN','Gaussian NB','BiLSTM','Temporal CNN','Gated posture + CNN']; auc=[e['mild']['posture_svm']['auc']['mean'],e['mild']['posture_extratrees']['auc']['mean'],e['mild']['posture_knn']['auc']['mean'],e['mild']['posture_gnb']['auc']['mean'],s['mild']['compact_bilstm']['auc'],s['mild']['temporal_cnn']['auc'],f['auc']]; sens=[e['mild']['posture_svm']['sensitivity']['mean'],e['mild']['posture_extratrees']['sensitivity']['mean'],e['mild']['posture_knn']['sensitivity']['mean'],e['mild']['posture_gnb']['sensitivity']['mean'],s['mild']['compact_bilstm']['sensitivity'],s['mild']['temporal_cnn']['sensitivity'],f['sensitivity']]
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True,layout='constrained'); order=np.arange(len(names)); colors=['#b35c28','#858585','#858585','#858585','#7970a8','#167d9a','#478260']
    axes[0].barh(order,auc,color=colors); axes[1].barh(order,sens,color=colors)
    for ax,val,title in [(axes[0],auc,'A  Cow-grouped ROC AUC'),(axes[1],sens,'B  Mild-case sensitivity')]:
        for i,v in enumerate(val):ax.text(v+.006,i,f'{v:.3f}',va='center',fontsize=8)
        ax.set(title=title,xlim=(.55,.94),xlabel='Performance'); ax.axvline(.5,color='black',ls=':',lw=.7)
    axes[0].set(yticks=order,yticklabels=names); axes[0].invert_yaxis(); fig.savefig(ROOT/'paper'/'figures'/'enhanced_models.png',dpi=300);fig.savefig(ROOT/'paper'/'figures'/'enhanced_models.pdf');plt.close(fig)
    sess=sorted([r for r in ep if r['validation']=='session_holdout' and r['model']=='posture_svm'],key=lambda r:(int(r['fold']),r['video'])); groups={}
    for r in sess:groups.setdefault(r['fold'],[]).append(r)
    labels=[];aucs=[];ns=[]
    for k,v in groups.items():labels.append(e['mild']['session_holdout']['sessions'][int(k)]['session']); yy=np.array([int(r['label']) for r in v]); ss=np.array([float(r['score']) for r in v]);aucs.append(roc_auc_score(yy,ss));ns.append(len(v))
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained'); ax.bar(labels,aucs,color='#167d9a');
    for i,(v,n) in enumerate(zip(aucs,ns)):ax.text(i,v+.015,f'{v:.2f}\nn={n}',ha='center',fontsize=8)
    ax.axhline(e['mild']['session_holdout']['posture_svm']['auc'],color='#b35c28',ls='--',label=f"Pooled AUC {e['mild']['session_holdout']['posture_svm']['auc']:.3f}");ax.set(ylim=(.45,1.04),xlabel='Released acquisition-session prefix',ylabel='ROC AUC',title='Strict same-farm acquisition-session holdout');ax.legend();fig.savefig(ROOT/'paper'/'figures'/'session_validation.png',dpi=300);fig.savefig(ROOT/'paper'/'figures'/'session_validation.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
