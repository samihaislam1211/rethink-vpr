"""Database augmentation common I/O and validated source-linked image catalogs. No inference on import."""
import gzip,hashlib,json,os,sys,uuid
from pathlib import Path
import numpy as np
from PIL import Image
from ..runtime import ROOT,SOURCE
DATABASE=ROOT/'database'
from ..generation import protocol as generation_protocol
from ..baselines import data as base
sha=base.sha;fp=base.fingerprint

def read(p):return json.loads(Path(p).read_text())
def atomic_bytes(path,data,exclusive=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  tmp.write_bytes(data)
  if exclusive:os.link(tmp,path)
  else:os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
def write(p,value,exclusive=False):atomic_bytes(p,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode(),exclusive)
def jsonl(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt') as f:return [json.loads(x) for x in f]
def write_jsonl(p,rows,exclusive=False):
 data=(''.join(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n' for x in rows)).encode()
 if str(p).endswith('.gz'):data=gzip.compress(data,mtime=0)
 atomic_bytes(p,data,exclusive)
def config(key):return next(x for x in read(DATABASE/'config/protocol.json')['experiments'] if x['key']==key)
def dataset(name):return next(x for x in read(ROOT/'baselines/config.json')['datasets'] if x['name']==name)
def real(name,role):return base.manifest(dataset(name),role)
def schedule(cfg):return read(ROOT/'generation/config/cameras.json')['schedules'][cfg['schedule']]
def catalog_path(arm,key):return DATABASE/'manifests'/f'{arm}__{key}.jsonl.gz'
def validate_catalog(rows,cfg,require_all=True):
 refs=real(cfg['dataset'],'reference');sources=read(DATABASE/'manifests'/f"sources__{cfg['key']}.json")['sources'];allowed={r['index']:r for r in sources};poses=schedule(cfg)['poses'];seen=set();images=set()
 for r in rows:
  pair=(r['source_index'],r['view_index'])
  if pair in seen or r['image_id'] in images:raise ValueError('duplicate source/view or image ID')
  seen.add(pair);images.add(r['image_id']);i,v=pair
  if i not in allowed or not 0<=v<8:raise ValueError('unselected source or invalid view')
  if r['source_role']!='reference' or r['source_id']!=refs[i]['id'] or r['source_sha256']!=allowed[i]['sha256']:raise ValueError('source/role/byte identity mismatch')
  if r['dataset']!=cfg['dataset'] or r['schedule']!=cfg['schedule'] or r['seed']!=generation_protocol.seed(i,v) or r['pose']!=poses[v]:raise ValueError('dataset/schedule/seed/pose mismatch')
  if not r['image_id'].startswith('synthetic/'+r['arm']+'/') or r['checkpoint_name']!='multi1':raise ValueError('invalid synthetic identity/checkpoint')
 if require_all and seen!={(s['index'],v) for s in sources for v in range(8)}:raise ValueError('incomplete source/view grid')
 return sorted(rows,key=lambda r:(r['source_index'],r['view_index']))
def check_image(row):
 info=generation_protocol.verify_png(row['path'],row.get('integrity',{}).get('sha256'))
 if 'integrity' in row and info!=row['integrity']:raise ValueError('image integrity metadata mismatch')
 return info

def current_env():
 import platform,torch,torchvision
 return {'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'cuda':torch.version.cuda}
def load_real(name,role,session=None):
 from ..generation import cache
 path=ROOT/'baselines/cache'/f"{name}__{'CosPlace'}__{role}.npy"
 metadata=read(path.with_suffix('.json'))
 return cache.load(path,metadata['provenance'],session=session,decision_dir=DATABASE/'evidence/real_cache_decisions')
