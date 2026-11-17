"""CosPlace synthetic extraction with separate, strictly validated Database augmentation caches."""
import argparse,fcntl,io,os,platform,socket,time
from pathlib import Path
import numpy as np
from . import common as c
from ..baselines import extract as extractor

def rows_for(arm,key,pilot=False):
 cfg=c.config(key)
 if arm!='fresh':raise ValueError('Only fresh generation is supported')
 rows=[c.read(p) for p in (c.DATABASE/'runs/fresh_v1'/key).glob('source_*/view_*.json')]
 rows=c.validate_catalog(rows,cfg,require_all=not pilot)
 if pilot:
  first=c.read(c.DATABASE/'manifests'/f'sources__{key}.json')['sources'][0]['index'];rows=[r for r in rows if r['source_index']==first and r['view_index']<2]
  if len(rows)!=2:raise ValueError('pilot needs two views of selected source')
 return rows

def cache_expected(rows,spec):
 return {'arm':rows[0]['arm'],'dataset':rows[0]['dataset'],'configuration':rows[0]['configuration'],'ordered_images_sha256':c.fp([(r['image_id'],r['source_id'],r['source_sha256'],r['view_index'],r['integrity']) for r in rows]),'model':spec,'preprocessing':c.read(c.ROOT/'baselines/cache'/f"{rows[0]['dataset']}__CosPlace__reference.json")['provenance']['preprocessing'],'normalization':'float32 L2 dim1 eps1e-12','environment':c.current_env(),'implementation_sha256':c.sha(__file__),'baseline_extractor_sha256':c.sha(c.SOURCE/'baselines/extract.py'),'database_module_sha256':c.sha(c.SOURCE/'database/common.py'),'seed':42}
def load_features(path,expected):
 m=c.read(path.with_suffix('.json'))
 if m['fingerprint']!=c.fp(expected) or m['provenance']!=expected or m['embedding_sha256']!=c.sha(path):raise ValueError('stale/incompatible/corrupt synthetic descriptor cache')
 a=np.load(path,mmap_mode='r')
 if list(a.shape)!=m['shape'] or a.shape[1]!=2048 or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5):raise ValueError('invalid synthetic descriptors')
 return a

def extract(arm,key,pilot=False):
 import torch,torchvision
 from PIL import Image
 if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Synthetic extraction requires Slurm CUDA')
 rows=rows_for(arm,key,pilot);suffix='__pilot' if pilot else '';path=c.DATABASE/'cache'/f'{arm}__{key}{suffix}.npy'
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 spec=extractor.model_spec('CosPlace');model=extractor.build(spec);spec['pool_p']=float(model.aggregation[1].p.detach().cpu());expected=cache_expected(rows,spec)
 original=c.read(c.ROOT/'baselines/cache'/f"{rows[0]['dataset']}__CosPlace__reference.json")['provenance']
 if spec!=original['model'] or c.current_env()!=original['environment']:raise ValueError('CosPlace differs from real-cache model/environment')
 with path.with_suffix('.lock').open('a') as lease:
  fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  if path.exists():load_features(path,expected);print('REUSED synthetic cache',path,flush=True);return
  if path.with_suffix('.json').exists():raise ValueError('orphan synthetic cache metadata')
  started=time.perf_counter();features=np.empty((len(rows),2048),dtype=np.float32)
  transform=torchvision.transforms.Compose([torchvision.transforms.Resize(480,interpolation=torchvision.transforms.InterpolationMode.BILINEAR,antialias=True),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);batch=[];indices=[]
  def flush():
   if not batch:return
   with torch.inference_mode():value=torch.nn.functional.normalize(model(torch.stack(batch).cuda()).float(),dim=1).cpu().numpy()
   if not np.isfinite(value).all():raise ValueError('nonfinite descriptors')
   features[indices]=value;batch.clear();indices.clear()
  for i,row in enumerate(rows):
   c.check_image(row)
   with Image.open(row['path']) as im:batch.append(transform(im.convert('RGB')))
   indices.append(i)
   if len(batch)==8:flush()
   if i%1000==0:print('EXTRACT',arm,key,i,'/',len(rows),flush=True)
  flush()
  if not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5):raise ValueError('descriptor normalization failed')
  buf=io.BytesIO();np.save(buf,features);c.atomic_bytes(path,buf.getvalue(),exclusive=True)
  c.write(path.with_suffix('.json'),{'fingerprint':c.fp(expected),'provenance':expected,'embedding_sha256':c.sha(path),'shape':list(features.shape),'seconds':time.perf_counter()-started,'execution':{'job_id':os.getenv('SLURM_JOB_ID'),'hostname':platform.node(),'gpu':torch.cuda.get_device_name(0),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated()}},exclusive=True)
  c.write_jsonl(c.DATABASE/'manifests'/f'{arm}__{key}{suffix}.jsonl.gz',rows)
  print('DESCRIPTORS_COMPLETE',arm,key,len(rows),flush=True)
