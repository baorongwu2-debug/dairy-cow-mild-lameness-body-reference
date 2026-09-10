"""Portable I/O compatibility layer for the published analysis scripts.

The unrelated exploratory video-classification pipeline is intentionally omitted.
"""
from pathlib import Path
import os, csv, json, hashlib
ROOT=Path(os.environ.get('LAMENESS_WORKDIR',Path(__file__).resolve().parents[1])).resolve()
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
