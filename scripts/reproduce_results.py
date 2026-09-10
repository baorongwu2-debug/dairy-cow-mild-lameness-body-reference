"""Verify the exact upstream data snapshot and run analysis in a separate workspace."""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',required=True,type=Path)
    ap.add_argument('--archive',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT/'reproduced')
    ap.add_argument('--stage',choices=['all','base','enhanced','sequence','fusion'],default='all')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args(); data=args.data_root.resolve(); dest=args.output.resolve()
    audit=json.loads((ROOT/'results/pose_results/data_audit.json').read_text())
    assert sha(data/'videos_lameness_scores.csv')==audit['label_sha256'],'Label checksum mismatch'
    with (ROOT/'results/pose_results/manifest.csv').open(encoding='utf-8-sig') as f: manifest=list(csv.DictReader(f))
    for r in manifest:
        assert sha(data/'videos_keypoints'/f"{r['video']}.csv")==r['sha256'],f"Trajectory checksum mismatch: {r['video']}"
    if args.archive: assert sha(args.archive)==audit['archive_sha256'],'ZIP checksum mismatch'
    print(f'Verified labels and {len(manifest)} trajectories.',flush=True)
    if args.check_only: return
    if dest==ROOT or dest==ROOT/'results' or (ROOT/'results') in dest.parents:
        raise ValueError('Use a separate reproduction directory, not the recorded evidence directory')
    stages=['base','enhanced','sequence','fusion'] if args.stage=='all' else [args.stage]
    folders={'base':'pose_results','enhanced':'enhanced_results','sequence':'sequence_results','fusion':'fusion_results'}
    for stage in stages:
        target=dest/folders[stage]
        if target.exists() and any(target.iterdir()): raise FileExistsError(f'{target} already contains output; choose another --output')
    dest.mkdir(parents=True,exist_ok=True)
    target=dest/'external_data/official/lstm-lameness-detection-main/data'
    for r in manifest:
        rel=Path('videos_keypoints')/f"{r['video']}.csv"; (target/rel).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(data/rel,target/rel)
    shutil.copy2(data/'videos_lameness_scores.csv',target/'videos_lameness_scores.csv')
    if args.archive: shutil.copy2(args.archive,dest/'external_data/russello_official.zip')
    for p in ['pose_protocol.md','enhanced_protocol.md']: shutil.copy2(ROOT/p,dest/p)
    env=os.environ.copy();env['LAMENESS_WORKDIR']=str(dest)
    files={'base':'pose_study.py','enhanced':'enhanced_study.py','sequence':'sequence_baselines.py','fusion':'temporal_fusion.py'}
    for stage in stages:
        if stage in ['enhanced','fusion'] and not (dest/'pose_results/features.npz').exists():
            raise FileNotFoundError('Run --stage base in the same output directory first')
        subprocess.run([sys.executable,str(ROOT/'src'/files[stage])],env=env,check=True,cwd=dest)
    if args.stage=='all':
        for p in ['finalize_pose.py','enhanced_finalize.py']:
            subprocess.run([sys.executable,str(ROOT/'src'/p)],env=env,check=True,cwd=dest)
    print('Reproduction outputs:',dest)
if __name__=='__main__': main()
