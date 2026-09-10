"""Independent evidence checks, full-data demonstration models, and paper figures."""
from pose_study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'body_combined_svm':'Body posture + motion SVM','image_combined_svm':'Image posture + motion SVM','body_static_svm':'Body posture SVM','body_dynamic_svm':'Body motion SVM','body_combined_lr':'Body combined logistic','body_combined_rf':'Body combined forest'}

def main():
    xs,ts,manifest=load_data(); z=np.load(OUT/'features.npz'); summary=json.loads((OUT/'summary.json').read_text()); splits=json.loads((OUT/'splits.json').read_text())
    with open(OUT/'predictions.csv',encoding='utf-8-sig') as f: predictions=list(csv.DictReader(f))
    assert len(predictions)==15330 and len(splits)==300
    cow=np.array([r['cow_id'] for r in manifest]); names=np.array([r['video'] for r in manifest]); labels=np.array([r['score'] for r in manifest])
    for s in splits:
        a=np.array(s['train']); b=np.array(s['test']); assert not set(cow[a])&set(cow[b]); assert not set(a)&set(b)
        for inner in s['inner']:
            tr=np.array(inner['train']); va=np.array(inner['validation']); assert not set(cow[tr])&set(cow[va]); assert set(tr)|set(va)==set(a)
    all_scores={}; extra={}; selected_dims={}
    for ep in ['mild','all']:
        ix=np.flatnonzero(labels<=2) if ep=='mild' else np.arange(272); y=(labels[ix]>1).astype(int); g=cow[ix]
        all_scores[ep]={}
        for name in LABELS:
            values=[]; all_scores[ep][name]=[]
            for seed in SEEDS:
                rows=[r for r in predictions if r['endpoint']==ep and r['model']==name and int(r['seed'])==seed]
                assert len(rows)==len(ix) and len(set(r['video'] for r in rows))==len(ix)
                lookup={r['video']:r for r in rows}
                assert all(lookup[names[j]]['cow_id']==cow[j] and int(lookup[names[j]]['label'])==int(labels[j]>1) for j in ix)
                sc=np.array([float(lookup[names[j]]['score']) for j in ix]); all_scores[ep][name].append(sc); values.append(measures(y,sc))
            for key in values[0]: np.testing.assert_allclose(np.mean([v[key] for v in values]),summary[ep][name][key]['mean'],atol=1e-12)
        s=next(s for s in splits if s['endpoint']==ep and s['seed']==42 and s['fold']==0 and s['model']=='body_combined_svm')
        model,_=classifier('svm'); model.set_params(**s['params']); model.fit(z['combined'][s['train']],(labels[s['train']]>1).astype(int))
        lookup={names[j]:i for i,j in enumerate(ix)}; actual=score(model,z['combined'][s['test']]); expected=all_scores[ep]['body_combined_svm'][0][[lookup[names[j]] for j in s['test']]]
        np.testing.assert_allclose(actual,expected,atol=1e-8)
        selected_dims[ep]=int(sum(model['variance'].get_support()))
        # Secondary geometry contrast computed from the retained seed-42 predictions.
        rng=np.random.default_rng(20260907); ids=np.unique(g); clusters={c:np.flatnonzero(g==c) for c in ids}; dif=[]
        for _ in range(2000):
            b=np.concatenate([clusters[c] for c in rng.choice(ids,len(ids),replace=True)])
            if len(set(y[b]))<2: continue
            dif.append(roc_auc_score(y[b],all_scores[ep]['body_combined_svm'][0][b])-roc_auc_score(y[b],all_scores[ep]['image_combined_svm'][0][b]))
        extra[ep]=dict(combined_body_minus_image=float(roc_auc_score(y,all_scores[ep]['body_combined_svm'][0])-roc_auc_score(y,all_scores[ep]['image_combined_svm'][0])),cow_bootstrap_interval=np.quantile(dif,[.025,.975]).tolist(),posture_seed42_confusion=confusion_matrix(y,all_scores[ep]['body_static_svm'][0]>=0).tolist())
        for variant,key in [('combined','combined'),('posture','static')]:
            clf,grid=classifier('svm'); search=GridSearchCV(clf,grid,cv=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(z[key][ix],y,g)),scoring='roc_auc',error_score='raise'); search.fit(z[key][ix],y)
            joblib.dump(search.best_estimator_,OUT/f'{ep}_{variant}_full_data.joblib')
    save_json(OUT/'secondary_contrasts.json',extra)
    save_json(OUT/'verification.json',dict(status='passed',prediction_rows=len(predictions),outer_folds=len(splits),inner_folds=sum(len(s['inner']) for s in splits),cow_overlap=0,all_mean_metrics_recomputed=True,two_saved_outer_folds_reproduced=True,retained_combined_features_example=selected_dims,upstream_pose_independence='not established',full_data_models='research demonstration only; no independent evaluation after full-data refit'))
    figs=ROOT/'paper'/'figures'; figs.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
    order=list(LABELS); colors=['#167d9a','#9b9b9b','#b35c28','#7970a8','#478260','#be7881']
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,ep in zip(axes,['mild','all']):
        for i,name in enumerate(order):
            mean=summary[ep][name]['auc']['mean']; sd=summary[ep][name]['auc']['split_sd']; ax.errorbar(mean,i,xerr=sd,fmt='o',color=colors[i],capsize=3)
        ax.set(xlim=(.6,.95),xlabel='ROC AUC, mean and SD across five split seeds',title='Normal versus mild' if ep=='mild' else 'Normal versus all lame scores',yticks=range(len(order)),yticklabels=[LABELS[n] for n in order]); ax.axvline(.5,color='gray',ls=':')
    axes[0].invert_yaxis(); fig.tight_layout(); fig.savefig(figs/'performance.png',dpi=300); fig.savefig(figs/'performance.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.7)); ep='mild'; y=(labels[labels<=2]>1).astype(int)
    for name,c in zip(order[:4],colors[:4]):
        sc=all_scores[ep][name][0]; fpr,tpr,_=roc_curve(y,sc); ax.plot(fpr,tpr,color=c,label=f'{LABELS[name]} ({roc_auc_score(y,sc):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=.7); ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Cow-grouped out-of-fold ROC, seed 42'); ax.legend(fontsize=8,loc='lower right'); fig.tight_layout(); fig.savefig(figs/'mild_roc.png',dpi=300); fig.savefig(figs/'mild_roc.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=xs[0][len(xs[0])//2]; c=coordinates(xs[0]).reshape(-1,9,2)[len(xs[0])//2]
    edges=[(4,5),(5,6),(6,8),(8,7),(6,0),(6,1),(7,2),(7,3)]
    for ax,arr,title in zip(axes,[x,c],['Released pose in image coordinates','Pose in the trunk reference frame']):
        for a,b in edges: ax.plot(arr[[a,b],0],arr[[a,b],1],color='#8c969e',lw=1)
        ax.scatter(arr[:,0],arr[:,1],c=['#167d9a']*4+['#b35c28']*2+['#478260']*3,s=28,zorder=3)
        for j,k in enumerate(KP): ax.annotate(k,(arr[j,0],arr[j,1]),xytext=(3,5),textcoords='offset points',fontsize=7)
        ax.invert_yaxis(); ax.set_aspect('equal'); ax.set_title(title)
    axes[0].set(xlabel='Image x, pixels',ylabel='Image y, pixels'); axes[1].set(xlabel='Trunk-normalized x',ylabel='Trunk-normalized y'); fig.tight_layout(); fig.savefig(figs/'reference_frame.png',dpi=300); fig.savefig(figs/'reference_frame.pdf'); plt.close(fig)
    print(json.dumps(dict(verification='passed',contrasts=extra),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
