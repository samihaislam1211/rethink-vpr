"""Application identities, atomic storage and query-side protocol. Import has no inference."""
import ast,gzip,hashlib,io,json,os,sys,uuid
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
from ..runtime import ROOT,SOURCE
QUERY=ROOT/'query'
from ..generation import protocol as generation_protocol
from ..baselines import data as base
sha=base.sha;fp=base.fingerprint
DATASETS={'Pittsburgh30k':(10000,6816),'Tokyo247':(75984,315)}
NULL=[['crop',{'keep':.90}],['crop',{'keep':.80}],['photometric',{'brightness':1.15,'contrast':1.,'color':1.}],['photometric',{'brightness':.87,'contrast':1.15,'color':1.}],['photometric',{'brightness':1.,'contrast':.88,'color':1.20}],['rotate',{'deg':5.}],['rotate',{'deg':-5.}],['crop_photometric',{'keep':.88,'brightness':1.10,'contrast':1.08}]]
GROUPS=[{'key':d+'__'+k,'dataset':d,'condition':k,'schedule':('near_zero_v1' if k=='zero' else ('pitts_realistic_v1' if d=='Pittsburgh30k' else 'tokyo_query_realistic_v1'))} for d in DATASETS for k in ['viewpoint','null','zero'] if k!='zero' or d=='Pittsburgh30k']
def read(p):return json.loads(Path(p).read_text())
def atomic_bytes(p,data,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with t.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  if exclusive:os.link(t,p)
  else:os.replace(t,p)
 finally:
  if t.exists():t.unlink()
def write(p,x,exclusive=False):atomic_bytes(p,(json.dumps(x,indent=2,allow_nan=False)+'\n').encode(),exclusive)
def jsonl(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return [json.loads(s) for s in f]
def write_jsonl(p,rows,exclusive=False):
 b=''.join(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n' for x in rows).encode()
 atomic_bytes(p,gzip.compress(b,mtime=0) if str(p).endswith('.gz') else b,exclusive)
def config(key):return next(e for e in GROUPS if e['key']==key)
def dataset(name):return next(d for d in read(ROOT/'baselines/config.json')['datasets'] if d['name']==name)
def real(ds,role):return base.manifest(dataset(ds),role)
def schedule(e):return read(ROOT/'generation/config/cameras.json')['schedules'][e['schedule']]
def failure_indices():return read(QUERY/'manifests/selected_failures.json')['indices']
def sources(e):
 rows=read(QUERY/'manifests'/('sources__'+e['dataset']+'.json'))['sources']
 return [s for s in rows if s['index'] in set(failure_indices())] if e['condition']=='zero' else rows
def path_for(e,i,v):return QUERY/'runs/fresh_v1'/e['key']/f'source_{i:06d}'/f'view_{v}.png'
def catalog(arm,key):return QUERY/'manifests'/f'{arm}__{key}.jsonl.gz'
def current_env():
 import platform,torch,torchvision
 return {'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'cuda':torch.version.cuda}
def check_image(row):
 p=Path(row['path']);old=row.get('integrity')
 if not p.is_file():raise ValueError('missing image '+str(p))
 with Image.open(p) as im:
  im.load()
  if im.format!='PNG' or im.mode!='RGB' or (row['condition']!='null' and im.size!=(512,512)):raise ValueError('invalid PNG')
  info={'sha256':sha(p),'bytes':p.stat().st_size,'width':im.width,'height':im.height,'format':'PNG','mode':'RGB'}
 if old is not None and old!=info:raise ValueError('image checksum/encoding mismatch')
 return info
def validate_rows(rows,e,complete=True):
 ss={s['index']:s for s in sources(e)};seen=set();images=set()
 for r in rows:
  i,v=r['source_index'],r['view_index'];pair=(i,v)
  if pair in seen or r['image_id'] in images:raise ValueError('duplicate source/view/image')
  seen.add(pair);images.add(r['image_id'])
  if i not in ss or not 0<=v<8:raise ValueError('unselected source/view')
  if any(r[k]!=val for k,val in {'dataset':e['dataset'],'configuration':e['key'],'condition':e['condition'],'source_role':'query','source_id':ss[i]['id'],'source_sha256':ss[i]['sha256'],'seed':generation_protocol.seed(i,v)}.items()):raise ValueError('source/protocol mapping mismatch')
  if r['arm'] not in ['fresh'] or r['image_id']!=f"synthetic/{r['arm']}/{e['key']}/{i}/{v}":raise ValueError('arm/identity mismatch')
  if e['condition']=='null':
   if r['transformation']!=NULL[v]:raise ValueError('null transform mismatch')
  elif r['pose']!=schedule(e)['poses'][v] or r['schedule']!=e['schedule'] or r['checkpoint_name']!='multi1':raise ValueError('pose/checkpoint mismatch')
 if complete and seen!={(i,v) for i in ss for v in range(8)}:raise ValueError('incomplete source/view grid')
 return sorted(rows,key=lambda r:(r['source_index'],r['view_index']))
def apply_null(img,kind,p,rng):
 # Conventional crop, photometric and rotation controls with source/view seeds.
 w,h=img.size
 if kind in ('crop','crop_photometric'):
  keep=p['keep'];cw,ch=int(w*keep),int(h*keep);x=rng.integers(0,w-cw+1);y=rng.integers(0,h-ch+1)
  img=img.crop((x,y,x+cw,y+ch)).resize((w,h),Image.BICUBIC)
 if kind in ('photometric','crop_photometric'):
  jitter=float(rng.uniform(.97,1.03))
  for attr,key in (('Brightness','brightness'),('Contrast','contrast'),('Color','color')):
   if key in p:img=getattr(ImageEnhance,attr)(img).enhance(p[key]*jitter)
 if kind=='rotate':img=img.rotate(p['deg']+float(rng.uniform(-.5,.5)),resample=Image.BICUBIC,expand=False)
 return img
def load_real(ds,role,session=None):
 from ..generation import cache
 path=ROOT/'baselines/cache'/f"{ds}__{'CosPlace'}__{role}.npy"
 metadata=read(path.with_suffix('.json'))
 return cache.load(path,metadata['provenance'],session=session,decision_dir=QUERY/'evidence/real_cache_decisions')
def top(scores,k=10):
 s=np.asarray(scores);ix=np.lexsort((np.arange(len(s)),-s))[:k];return ix.tolist(),s[ix].tolist()
def full_fusion(similarity):
 s=np.asarray(similarity)
 if s.ndim!=2 or s.shape[0]!=9 or not np.isfinite(s).all():raise ValueError('need original plus eight finite view rows')
 return top(s.max(axis=0))
def portable_fusion(view_top10):
 if len(view_top10)!=9:raise ValueError('missing original or synthetic views')
 values={}
 for v in view_top10:
  ix,sc=v['indices'],v['scores']
  if len(ix)!=10 or len(sc)!=10 or len(set(ix))!=10 or not np.isfinite(sc).all():raise ValueError('invalid top10 evidence')
  if list(zip(ix,sc))!=sorted(zip(ix,sc),key=lambda z:(-z[1],z[0])):raise ValueError('noncanonical view ranking')
  for i,s in zip(ix,sc):values[i]=max(values.get(i,-float('inf')),s)
 pairs=sorted(values.items(),key=lambda z:(-z[1],z[0]))[:10];return [x[0] for x in pairs],[x[1] for x in pairs]
def paired(b,a,seed=42,reps=10000):
 b=np.asarray(b,dtype=bool);a=np.asarray(a,dtype=bool)
 if b.shape!=a.shape or not len(a):raise ValueError('paired population mismatch')
 gain=int((~b&a).sum());loss=int((b&~a).sum());n=len(a);d=gain+loss
 import math
 p=min(1.,2*sum(math.comb(d,j) for j in range(min(gain,loss)+1))/2**d) if d else 1.
 # Multinomial resampling of paired query differences is exactly the paired bootstrap distribution.
 draws=np.random.default_rng(seed).multinomial(n,[loss/n,(n-loss-gain)/n,gain/n],size=reps)
 delta=(draws[:,2]-draws[:,0])/n;lo,hi=np.quantile(delta,[.025,.975])
 return {'n':n,'gains':gain,'losses':loss,'retained_correct':int((b&a).sum()),'retained_incorrect':int((~b&~a).sum()),'net_recall1':(gain-loss)/n,'mcnemar_exact_two_sided':p,'bootstrap':{'seed':seed,'replicates':reps,'method':'paired query differences; equivalent multinomial counts (-1,0,+1); percentile95','CI95':[float(lo),float(hi)]}}
