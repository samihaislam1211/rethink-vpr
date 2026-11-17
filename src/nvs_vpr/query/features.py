"""Whole-query sharded CosPlace extraction; eight-view groups never straddle shards."""
import argparse,fcntl,io,os,platform,time
from pathlib import Path
import numpy as np
from PIL import Image
from . import common as c
from ..baselines import extract as extractor
def fresh_row(e,s,v):
 return {'arm':'fresh','dataset':e['dataset'],'configuration':e['key'],'condition':e['condition'],'source_role':'query','source_index':s['index'],'source_id':s['id'],'source_sha256':s['sha256'],'source_path':s['path'],'view_index':v,'seed':c.generation_protocol.seed(s['index'],v),'path':str(c.path_for(e,s['index'],v)),'image_id':f"synthetic/fresh/{e['key']}/{s['index']}/{v}",'transformation':c.NULL[v],'implementation_sha256':c.sha(__file__),'query_module_sha256':c.sha(c.SOURCE/'query/common.py')}
def nulls(shard,shards):
 for e in c.GROUPS:
  if e['condition']!='null':continue
  for s in c.sources(e):
   if s['index']%shards!=shard:continue
   if c.sha(s['path'])!=s['sha256']:raise ValueError('null source changed')
   with Image.open(s['path']) as im:source=im.convert('RGB')
   for v in range(8):
    row=fresh_row(e,s,v);p=Path(row['path']);meta=p.with_suffix('.json');p.parent.mkdir(parents=True,exist_ok=True)
    if meta.exists():
     saved=c.read(meta)
     if any(saved.get(k)!=val for k,val in row.items()):raise ValueError('null resume provenance mismatch')
     c.check_image(saved);continue
    if p.exists():raise ValueError('orphan null image; no overwrite')
    result=c.apply_null(source.copy(),*c.NULL[v],np.random.default_rng(row['seed']));buf=io.BytesIO();result.save(buf,format='PNG');c.atomic_bytes(p,buf.getvalue(),exclusive=True)
    row['integrity']=c.check_image(row);row['Slurm_job_id']=os.getenv('SLURM_JOB_ID');c.write(meta,row,exclusive=True)
   if s['index']%100==shard:print('NULL_SOURCE',e['key'],s['index'],flush=True)
def rows_for(arm,e,pilot=False,shard=None,shards=12):
 if arm!='fresh':raise ValueError('Only fresh generation is supported')
 if arm=='fresh':
  rows=[]
  ss=c.sources(e)
  for s in (ss[:1] if pilot else ss):
   if shard is not None and s['index']%shards!=shard:continue
   for v in range(2 if pilot else 8):rows.append(c.read(c.path_for(e,s['index'],v).with_suffix('.json')))
 rows=c.validate_rows(rows,e,complete=not pilot and shard is None)
 if pilot:rows=[r for r in rows if r['source_index']==c.sources(e)[0]['index'] and r['view_index']<2]
 return rows
def expected(rows,spec):
 return {'arm':rows[0]['arm'],'group':rows[0]['configuration'],'ordered_images_sha256':c.fp([(r['image_id'],r['source_id'],r['source_sha256'],r['view_index'],r['integrity']) for r in rows]),'model':spec,'environment':c.current_env(),'preprocessing':c.read(c.ROOT/'baselines/cache'/f"{rows[0]['dataset']}__CosPlace__reference.json")['provenance']['preprocessing'],'normalization':'float32 L2 eps1e-12','implementation_sha256':c.sha(__file__),'baseline_extractor_sha256':c.sha(c.SOURCE/'baselines/extract.py'),'query_module_sha256':c.sha(c.SOURCE/'query/common.py'),'seed':42}
def load_features(path,ex):
 meta=c.read(path.with_suffix('.json'))
 if meta['fingerprint']!=c.fp(ex) or meta['provenance']!=ex or meta['embedding_sha256']!=c.sha(path):raise ValueError('stale/corrupt descriptor cache')
 a=np.load(path,mmap_mode='r')
 if list(a.shape)!=meta['shape'] or a.shape[1]!=2048 or a.dtype!=np.float32 or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5):raise ValueError('invalid feature values/shape')
 return a
def shard_path(arm,key,shard,pilot=False):return c.QUERY/'cache'/f"{arm}__{key}__{'pilot' if pilot else f'shard{shard:02d}'}.npy"
def extract(arm,shard,shards,pilot=False):
 import torch,torchvision
 if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('needs Slurm CUDA')
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 spec=extractor.model_spec('CosPlace');model=extractor.build(spec);spec['pool_p']=float(model.aggregation[1].p.detach().cpu())
 original=c.read(c.ROOT/'baselines/cache/Pittsburgh30k__CosPlace__reference.json')['provenance']
 if spec!=original['model'] or c.current_env()!=original['environment']:raise ValueError('model/environment differs from baseline')
 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(480,interpolation=torchvision.transforms.InterpolationMode.BILINEAR,antialias=True),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 for e in c.GROUPS:
  if pilot and e['condition']=='null':continue
  rows=rows_for(arm,e,pilot,shard=None if pilot else shard,shards=shards);rows=[r for r in rows if pilot or r['source_index']%shards==shard]
  if not rows:continue
  path=shard_path(arm,e['key'],shard,pilot);ex=expected(rows,spec)
  if path.exists():load_features(path,ex);continue
  if path.with_suffix('.json').exists():raise ValueError('orphan feature metadata')
  t=time.time();values=np.empty((len(rows),2048),dtype=np.float32);batch=[];indices=[]
  def flush():
   if not batch:return
   with torch.inference_mode():v=torch.nn.functional.normalize(model(torch.stack(batch).cuda()).float(),dim=1,eps=1e-12).cpu().numpy()
   if not np.isfinite(v).all():raise ValueError('nonfinite CosPlace output')
   values[indices]=v;batch.clear();indices.clear()
  for i,r in enumerate(rows):
   c.check_image(r)
   with Image.open(r['path']) as im:im=transform(im.convert('RGB'))
   if batch and (len(batch)==8 or im.shape!=batch[0].shape):flush()
   batch.append(im);indices.append(i)
   if i%2000==0:print('DESCRIPTOR',arm,e['key'],i,len(rows),flush=True)
  flush();buf=io.BytesIO();np.save(buf,values);c.atomic_bytes(path,buf.getvalue(),exclusive=True)
  c.write(path.with_suffix('.json'),{'fingerprint':c.fp(ex),'provenance':ex,'embedding_sha256':c.sha(path),'shape':list(values.shape),'seconds':time.time()-t,'job_id':os.getenv('SLURM_JOB_ID'),'array_job_id':os.getenv('SLURM_ARRAY_JOB_ID'),'gpu':torch.cuda.get_device_name(),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated()},exclusive=True)
  c.write_jsonl(c.QUERY/'manifests'/path.with_suffix('.jsonl.gz').name,rows);load_features(path,ex)
def merge(arm,shards):
 for e in c.GROUPS:
  rows=rows_for(arm,e);allrows=[];features=[]
  for shard in range(shards):
   rr=[r for r in rows if r['source_index']%shards==shard]
   if not rr:continue
   p=shard_path(arm,e['key'],shard);spec=c.read(p.with_suffix('.json'))['provenance']['model'];features.append(np.asarray(load_features(p,expected(rr,spec))));allrows.extend(rr)
  order=sorted(range(len(allrows)),key=lambda j:(allrows[j]['source_index'],allrows[j]['view_index']));values=np.concatenate(features)[order];canonical=[allrows[j] for j in order]
  if canonical!=rows:raise ValueError('feature merge order mismatch')
  c.write_jsonl(c.catalog(arm,e['key']),rows)
  p=c.QUERY/'cache'/f'{arm}__{e["key"]}.npy';ex=expected(rows,spec)
  if p.exists():load_features(p,ex)
  else:
   buf=io.BytesIO();np.save(buf,values);c.atomic_bytes(p,buf.getvalue(),exclusive=True);c.write(p.with_suffix('.json'),{'fingerprint':c.fp(ex),'provenance':ex,'embedding_sha256':c.sha(p),'shape':list(values.shape)},exclusive=True)
 c.write(c.QUERY/'evidence'/f'descriptors_{arm}.json',{'status':'pass','groups':len(c.GROUPS),'rows':sum(len(c.sources(e))*8 for e in c.GROUPS)})
def pilot():
 from . import retrieval as query_retrieval
 extract('fresh',0,1,pilot=True);checks=[]
 for e in c.GROUPS:
  if e['condition']=='null':continue
  rr=rows_for('fresh',e,True);p=shard_path('fresh',e['key'],0,True);m=c.read(p.with_suffix('.json'));a=load_features(p,m['provenance']);db=c.load_real(e['dataset'],'reference');q=c.load_real(e['dataset'],'query');i=rr[0]['source_index'];views=np.vstack([q[i],a]);ix,sc=c.base.topk(db,views)
  # A small complete nine-view matrix validates true max fusion independently in smoke.
  matrix=np.vstack([q[i],a,a,a,a])@np.asarray(db[:32]).T
  from .ranking import union_top
  tops=[dict(indices=c.top(v)[0],scores=c.top(v)[1]) for v in matrix]
  if union_top(tops)!=c.full_fusion(matrix):raise ValueError('pilot portable fusion mismatch')
  checks.append({'configuration':e['key'],'source_id':rr[0]['source_id'],'finite_unit_descriptors':True,'real_references_only':len(db),'view_top1':ix[:,0].tolist(),'small_complete_matrix_match':True})
 c.write(c.QUERY/'evidence/pilot_end_to_end.json',{'passed':True,'checks':checks,'job_id':os.getenv('SLURM_JOB_ID')})
