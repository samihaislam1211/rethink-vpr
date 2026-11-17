"""Application isolated storage, canonical inputs and strict descriptor provenance."""
import functools,io,os,sys
from pathlib import Path
import numpy as np
from ..runtime import ROOT,SOURCE
ANALYSIS=ROOT/'analysis'
from ..query import common as query_core
read=query_core.read;write=query_core.write;atomic_bytes=query_core.atomic_bytes;jsonl=query_core.jsonl;write_jsonl=query_core.write_jsonl;sha=query_core.sha;fp=query_core.fp
DATASETS=query_core.DATASETS;MODELS=['CosPlace','EigenPlaces','AlexNet-GeM','ResNet50-GeM'];NEW_MODELS=MODELS[1:];FUSIONS=['max','mean','conf'];SIGNALS=['margin','top1','entropy'];ARMS=['fresh'];CONDITIONS=['viewpoint','null']
TASKS=[dict(arm=a,dataset=d,model=m) for a in ARMS for d in DATASETS for m in MODELS]
def task(i):return TASKS[i]
def key(t,condition=None):return '__'.join([t['arm'],t['dataset'],t['model']]+([condition] if condition else []))
def real(ds,role):return query_core.real(ds,role)
def require_slurm():
 if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('long computation requires Slurm allocation')
def load_real(ds,role,model,session=None):
 from ..generation import cache
 path=ROOT/'baselines/cache'/f"{ds}__{model}__{role}.npy"
 metadata=read(path.with_suffix('.json'))
 return cache.load(path,metadata['provenance'],session=session,decision_dir=ANALYSIS/'evidence/real_cache_decisions')
def catalog(arm,ds,condition):return ANALYSIS/'manifests'/f'{arm}__{ds}__{condition}.jsonl.gz'
def rows_for(arm,ds,condition):
 rows=jsonl(catalog(arm,ds,condition));query_core.validate_rows(rows,query_core.config(ds+'__'+condition))
 if any(r['arm']!=arm for r in rows):raise ValueError('cross-arm image catalog')
 return rows
def feature_path(arm,ds,model,condition,shard=None):return ANALYSIS/'cache'/f"{arm}__{ds}__{model}__{condition}{'' if shard is None else f'__shard{shard}'}.npy"
def ordered_hash(rows):return fp([(r['image_id'],r['source_id'],r['source_sha256'],r['view_index'],r['integrity']) for r in rows])
def feature_expected(rows,model):
 ds=rows[0]['dataset'];realmeta=read(ROOT/'baselines/cache'/f'{ds}__{model}__reference.json')['provenance']
 return {'arm':rows[0]['arm'],'dataset':ds,'model':realmeta['model'],'condition':rows[0]['condition'],'ordered_images_sha256':ordered_hash(rows),'preprocessing':realmeta['preprocessing'],'normalization':realmeta['normalization'],'environment':query_core.current_env(),'seed':42,'extractor_sha256':sha(SOURCE/'analysis/features.py'),'analysis_module_sha256':sha(__file__),'runner_sha256':sha(SOURCE/'baselines/extract.py')}
def checked_features(path,expected):
 meta=read(path.with_suffix('.json'))
 if meta['fingerprint']!=fp(expected) or meta['provenance']!=expected or meta['embedding_sha256']!=sha(path):raise ValueError('descriptor cache fingerprint/checksum mismatch')
 a=np.load(path,mmap_mode='r');dim=expected['model']['dim']
 if list(a.shape)!=meta['shape'] or a.shape[1]!=dim or a.dtype!=np.float32:raise ValueError('feature dimensions/dtype mismatch')
 for k in range(0,len(a),1024):
  z=a[k:k+1024]
  if not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5):raise ValueError('invalid feature normalization/values')
 return a
def load_synthetic(t,condition):
 rows=rows_for(t['arm'],t['dataset'],condition);model=t['model']
 if model=='CosPlace':
  from ..query import features as query_features
  p=ROOT/'query/cache'/f"{t['arm']}__{t['dataset']}__{condition}.npy";meta=read(p.with_suffix('.json'));a=query_features.load_features(p,query_features.expected(rows,meta['provenance']['model']))
  if meta['provenance']['model']!=read(ROOT/'baselines/cache'/f"{t['dataset']}__CosPlace__reference.json")['provenance']['model']:raise ValueError('legacy descriptor identity differs')
  write(ANALYSIS/'evidence/cosplace_reuse'/f'{key(t,condition)}.json',{'accepted':True,'unchanged_cache_path':str(p),'embedding_sha256':sha(p),'metadata_sha256':sha(p.with_suffix('.json')),'ordered_images_sha256':ordered_hash(rows),'query_extractor_sha256':sha(SOURCE/'query/features.py'),'query_fingerprint':meta['fingerprint']})
  return rows,a,p
 p=feature_path(t['arm'],t['dataset'],model,condition);return rows,checked_features(p,feature_expected(rows,model)),p
def baseline(ds,model):return jsonl(ANALYSIS/'evidence'/f'baseline__{ds}__{model}.jsonl.gz')
def gt(ds):
 with np.load(ANALYSIS/'manifests'/f'GT__{ds}.npz',allow_pickle=False) as z:return {name:[z[name+'_indices'][z[name+'_ptr'][i]:z[name+'_ptr'][i+1]].tolist() for i in range(DATASETS[ds][1])] for name in ['radius25m','radius50m']}
def save_array(path,array,exclusive=True):
 buf=io.BytesIO();np.save(buf,array);atomic_bytes(path,buf.getvalue(),exclusive)
def save_npz(path,**arrays):
 buf=io.BytesIO();np.savez_compressed(buf,**arrays);atomic_bytes(path,buf.getvalue())
