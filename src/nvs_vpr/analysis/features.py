"""Additional frozen descriptors only. No image-generation code path."""
import argparse,io,os,platform,socket,time
from pathlib import Path
import numpy as np
from PIL import Image
from . import common as c
from ..baselines import extract as extractor
def configure(model_name):
 import torch,torchvision
 c.require_slurm()
 if not torch.cuda.is_available():raise RuntimeError('needs scheduler-allocated CUDA')
 def offline(*a,**k):raise RuntimeError('Robustness analysis model downloads/network disabled')
 torch.hub.load=torch.hub.load_state_dict_from_url=torch.hub.download_url_to_file=offline;socket.socket.connect=offline
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 spec=extractor.model_spec(model_name);model=extractor.build(spec);spec['pool_p']=float(model.aggregation[1].p.detach().cpu()) if model_name in ['CosPlace','EigenPlaces'] else 3.
 original=c.read(c.ROOT/'baselines/cache'/f'Pittsburgh30k__{model_name}__reference.json')['provenance']
 if spec!=original['model'] or c.query_core.current_env()!=original['environment'] or model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('descriptor identity/environment/mode mismatch')
 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(480,interpolation=torchvision.transforms.InterpolationMode.BILINEAR,antialias=True),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 return model,spec,transform
def encode(rows,model,transform,spec):
 import torch
 values=np.empty((len(rows),spec['dim']),np.float32);batch=[];ids=[]
 def flush():
  if not batch:return
  with torch.inference_mode():z=torch.nn.functional.normalize(model(torch.stack(batch).cuda()).float(),dim=1,eps=1e-12).cpu().numpy()
  if z.shape!=(len(batch),spec['dim']) or not np.isfinite(z).all():raise ValueError('bad inference tensor')
  values[ids]=z;batch.clear();ids.clear()
 for i,r in enumerate(rows):
  c.query_core.check_image(r)
  with Image.open(r['path']) as im:im=transform(im.convert('RGB'))
  if batch and (len(batch)==8 or im.shape!=batch[0].shape):flush()
  batch.append(im);ids.append(i)
  if i%2000==0:print('EXTRACT',spec['name'],i,len(rows),flush=True)
 flush()
 if not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5):raise ValueError('descriptor normalization failed')
 return values
def extract(index):
 if not 0<=index<18:raise ValueError('invalid extraction shard')
 model_name=c.NEW_MODELS[index//6];arm=c.ARMS[0];shard=index%6;model,spec,transform=configure(model_name)
 for ds in c.DATASETS:
  for condition in c.CONDITIONS:
   rows=[r for r in c.rows_for(arm,ds,condition) if r['source_index']%6==shard];p=c.feature_path(arm,ds,model_name,condition,shard);ex=c.feature_expected(rows,model_name)
   if p.exists():c.checked_features(p,ex);print('REUSED',p.name,flush=True);continue
   if p.with_suffix('.json').exists():raise ValueError('orphan feature metadata')
   start=time.perf_counter();values=encode(rows,model,transform,spec);c.save_array(p,values)
   c.write(p.with_suffix('.json'),{'fingerprint':c.fp(ex),'provenance':ex,'embedding_sha256':c.sha(p),'shape':list(values.shape),'seconds':time.perf_counter()-start,'job_id':os.getenv('SLURM_JOB_ID'),'array_job_id':os.getenv('SLURM_ARRAY_JOB_ID'),'gpu':__import__('torch').cuda.get_device_name(),'model_eval':not model.training,'all_parameters_frozen':True},exclusive=True)
   c.checked_features(p,ex)
def merge(index):
 model=c.NEW_MODELS[index];arm=c.ARMS[0]
 for ds in c.DATASETS:
  for condition in c.CONDITIONS:
   rows=c.rows_for(arm,ds,condition);pieces=[];indices=[]
   for shard in range(6):
    selected=[j for j,r in enumerate(rows) if r['source_index']%6==shard];rr=[rows[j] for j in selected];p=c.feature_path(arm,ds,model,condition,shard);pieces.append(np.asarray(c.checked_features(p,c.feature_expected(rr,model))));indices.extend(selected)
   values=np.concatenate(pieces)[np.argsort(indices)];p=c.feature_path(arm,ds,model,condition);ex=c.feature_expected(rows,model)
   if p.exists():c.checked_features(p,ex);continue
   c.save_array(p,values);c.write(p.with_suffix('.json'),{'fingerprint':c.fp(ex),'provenance':ex,'embedding_sha256':c.sha(p),'shape':list(values.shape),'merged_from':6,'job_id':os.getenv('SLURM_JOB_ID')},exclusive=True);c.checked_features(p,ex)
 c.write(c.ANALYSIS/'evidence/merged'/f'{arm}__{model}.json',{'status':'pass'})
