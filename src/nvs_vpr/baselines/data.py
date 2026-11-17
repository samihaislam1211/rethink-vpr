"""Pure CPU data/GT/retrieval utilities; no existing loaders imported."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from ..runtime import ROOT,SOURCE
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def manifest(ds,role):return rows(ROOT/'manifests'/ds[role])
def positives(ds):
 r,q=manifest(ds,'reference'),manifest(ds,'query')
 if ds['name']=='GardensPoint':
  return {'exact':[[i] for i in range(200)],'window8':[list(range(max(0,i-8),min(200,i+9))) for i in range(200)]}
 if ds['name']=='StLucia':
  with np.load(ds['gt_file'],allow_pickle=False) as z:
   return {k:[np.flatnonzero(z[n][:,j]).tolist() for j in range(len(q))] for k,n in [('hard','GThard'),('soft','GTsoft')]}
 xy=lambda rows:np.array([[float(t) for t in x['filename'].split('@')[1:3]] for x in rows])
 t=cKDTree(xy(r));return {f'radius{radius}m': [sorted(a) for a in t.query_ball_point(xy(q),radius)] for radius in [25,50]}
def topk(db,queries,k=10,chunk=4096):
 """Cosine on normalized features, ties by ascending original ref index."""
 out=[];scores=[];k=min(k,len(db))
 for q in queries:
  ix=np.empty(0,dtype=int);sc=np.empty(0,dtype=np.float32)
  for start in range(0,len(db),chunk):
   s=np.asarray(db[start:start+chunk])@q
   candidates=np.concatenate([ix,np.arange(start,start+len(s))]);values=np.concatenate([sc,s])
   order=np.lexsort((candidates,-values))[:k];ix,sc=candidates[order],values[order]
  out.append(ix);scores.append(sc)
 return np.array(out),np.array(scores)
def metrics(pred,gt):
 valid=[j for j,g in enumerate(gt) if g]
 flags={k:[bool(set(pred[j,:k])&set(gt[j])) if gt[j] else None for j in range(len(gt))] for k in [1,5,10]}
 return {'valid_queries':len(valid),'excluded_queries':len(gt)-len(valid),'numerators':{str(k):sum(flags[k][j] for j in valid) for k in flags},'recall':{str(k):sum(flags[k][j] for j in valid)/len(valid) if valid else None for k in flags},'flags':flags}
def cache_meta(ds,role,model,environment):
 data=manifest(ds,role)
 return {'dataset':ds['name'],'role':role,'manifest_sha256':sha(ROOT/'manifests'/ds[role]),'image_stats_sha256':fingerprint([(x['id'],os.stat(x['path']).st_size,os.stat(x['path']).st_mtime_ns) for x in data]),'model':model,'preprocessing':'RGB Resize(shorter=480,bilinear,antialias=True) ToTensor ImageNet mean/std; no crop','normalization':'float32 L2 dim1 eps1e-12','environment':environment,'seed':42,'implementation_sha256':sha(SOURCE/'baselines/extract.py'),'core_sha256':sha(__file__)}
def load_cache(p,expected):
 meta=json.loads(p.with_suffix('.json').read_text())
 if meta['fingerprint']!=fingerprint(expected) or meta['embedding_sha256']!=sha(p):raise ValueError('stale/incompatible/corrupt cache')
 a=np.load(p,mmap_mode='r')
 if not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5):raise ValueError('invalid descriptors')
 return a
