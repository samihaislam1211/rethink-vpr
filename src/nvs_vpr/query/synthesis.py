"""Source-sharded Query expansion generation using unchanged Generation local model/camera/PNG adapter."""
import argparse,fcntl,hashlib,importlib.metadata as md,json,os,platform,random,socket,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from . import common as c
from ..generation import engine as generator

def shard_for(source_id,configuration,total):return int(hashlib.sha256((configuration+'|'+source_id).encode()).hexdigest()[:16],16)%total

def expected_row(exp,source,view,signature):
 i=source['index'];return {'arm':'fresh','dataset':exp['dataset'],'configuration':exp['key'],'source_role':'query','condition':exp['condition'],'source_index':i,'source_id':source['id'],'source_path':source['path'],'source_sha256':source['sha256'],'view_index':view,'schedule':exp['schedule'],'pose':c.schedule(exp)['poses'][view],'seed':c.generation_protocol.seed(i,view),'depth_seed':c.generation_protocol.seed(i,0),'checkpoint_name':'multi1','preprocessing':c.schedule(exp)['preprocessing'],'path':str(c.QUERY/'runs/fresh_v1'/exp['key']/f'source_{i:06d}'/f'view_{view}.png'),'image_id':f"synthetic/fresh/{exp['key']}/{i}/{view}",'generation_fingerprint':signature,'depth_policy':'once per source; fixed source-view0 seed; validated exact repeated depth in pilot'}

def resume_row(row):
 p=Path(row['path']);meta=p.with_suffix('.json')
 if meta.exists():
  saved=c.read(meta)
  if any(saved.get(k)!=v for k,v in row.items()):raise ValueError('resume provenance mismatch')
  c.check_image(saved);return saved
 if p.exists():raise ValueError('orphan/untracked PNG; refusing overwrite')
 return None

def run(a):
 if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('GPU generation requires Slurm')
 import torch
 from torchvision.transforms.functional import to_tensor,to_pil_image
 if not torch.cuda.is_available():raise RuntimeError('No allocated CUDA')
 if not c.read(c.QUERY/'evidence/preflight.json')['status']=='pass':raise ValueError('preflight not passed')
 if not a.pilot and not c.read(c.QUERY/'evidence/pilot_end_to_end.json')['passed']:raise ValueError('end-to-end pilot not passed')
 cfg=c.read(c.ROOT/'generation/config/genwarp.json');originals=c.read(c.ROOT/'generation/evidence/genwarp_originals.json');experiments=[e for e in c.GROUPS if e['condition']!='null']
 sources={e['key']:c.sources(e) for e in experiments}
 fingerprint=c.fp({'version':'query_fresh_v1','model':cfg,'originals':originals,'schedules':{e['key']:c.schedule(e) for e in experiments},'source_lists':{k:[(s['id'],s['sha256']) for s in ss] for k,ss in sources.items()},'generation_source_sha256':c.sha(__file__),'generation_adapter_sha256':c.sha(c.SOURCE/'generation/engine.py'),'common_sha256':c.sha(c.SOURCE/'generation/protocol.py'),'query_module_sha256':c.sha(c.SOURCE/'query/common.py'),'depth_policy':'once per source; source-view0 depth seed; exact repeated depth pilot'})
 tasks=[]
 for exp in experiments:
  for source in sources[exp['key']]:
   if a.pilot and source!=sources[exp['key']][0]:continue
   if not a.pilot and shard_for(source['id'],exp['key'],a.shards)!=a.shard:continue
   views=list(range(2 if a.pilot else 8));rows=[expected_row(exp,source,v,fingerprint) for v in views]
   if not all(resume_row(r) is not None for r in rows):tasks.append((exp,source,rows))
 if not tasks:print('COMPLETE no pending generation; no model loaded',flush=True);return
 for path,before in originals.items():
  if c.sha(path)!=before['sha256']:raise ValueError('original source/checkpoint changed '+path)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.hub.load=generator.deny_download;torch.hub.load_state_dict_from_url=generator.deny_download;torch.hub.download_url_to_file=generator.deny_download;socket.socket.connect=generator.deny_download
 job=os.getenv('SLURM_JOB_ID');run_id=('pilot' if a.pilot else f'shard{a.shard:02d}')+'_'+job;runpath=c.QUERY/'runs/generation_jobs'/run_id;runpath.mkdir(parents=True,exist_ok=False)
 torch.manual_seed(42);started=time.perf_counter();nvs,zoe,matrices,crop=generator.local_models(cfg);loaded=time.perf_counter()-started
 c.write(runpath/'run.json',{'run_id':run_id,'generation_fingerprint':fingerprint,'model':cfg,'checkpoint_and_source_hashes':originals,'adapter_sha256':c.sha(__file__),'query_module_sha256':c.sha(c.SOURCE/'query/common.py'),'generation_adapter_sha256':c.sha(c.SOURCE/'generation/engine.py'),'job_id':job,'array_job_id':os.getenv('SLURM_ARRAY_JOB_ID'),'array_task_id':os.getenv('SLURM_ARRAY_TASK_ID'),'hostname':platform.node(),'gpu':torch.cuda.get_device_name(0),'environment':{'python':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,'packages':{p:md.version(p) for p in ['torchvision','diffusers','transformers','huggingface_hub','timm','numpy','Pillow','splatting']}},'scheduler':dict(nvs.scheduler.config),'timesteps':nvs.scheduler.timesteps.cpu().tolist(),'all_modules_eval':all(not getattr(nvs,x).training for x in ['vae','image_encoder','reference_unet','denoising_unet','pose_guider']) and not zoe.training,'network_disabled':True,'model_loading_seconds':loaded,'pilot':a.pilot,'shard':a.shard,'shards':a.shards})
 depth_tests=[];generated=0
 for exp,source,rows in tasks:
  directory=Path(rows[0]['path']).parent;directory.mkdir(parents=True,exist_ok=True)
  with (directory/'.lock').open('a') as lease:
   fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   pending=[r for r in rows if resume_row(r) is None]
   if not pending:continue
   if c.sha(source['path'])!=source['sha256']:raise ValueError('original source bytes changed')
   pil=crop(source['path'],512) if c.schedule(exp)['preprocessing']=='center_square_lanczos512' else Image.open(source['path']).convert('RGB')
   src32=to_tensor(pil)[None].cuda().float();source_half=src32.half();torch.cuda.synchronize();depth_start=time.perf_counter()
   depth_seed=c.generation_protocol.seed(source['index'],0);torch.manual_seed(depth_seed);random.seed(depth_seed);np.random.seed(depth_seed)
   with torch.no_grad():depth=zoe.infer(src32,pad_input=True,with_flip_aug=True).half()
   torch.cuda.synchronize();depth_seconds=time.perf_counter()-depth_start
   if not torch.isfinite(depth).all() or (depth<=0).any():raise ValueError('invalid depth')
   if a.pilot:
    torch.manual_seed(depth_seed);random.seed(depth_seed);np.random.seed(depth_seed)
    with torch.no_grad():repeated=zoe.infer(src32,pad_input=True,with_flip_aug=True).half()
    identical=torch.equal(depth,repeated);depth_tests.append({'configuration':exp['key'],'source_id':source['id'],'exact_repeated_depth':identical,'max_absolute_difference':float((depth-repeated).abs().max()),'seed':depth_seed})
    if not identical:raise ValueError('Depth reuse changes numerical input; stop before bulk generation')
    del repeated
   mean=float(depth.mean().cpu());depth_hash=hashlib.sha256(depth.cpu().numpy().tobytes()).hexdigest()
   for row in pending:
    torch.cuda.reset_peak_memory_stats();torch.manual_seed(row['seed']);random.seed(row['seed']);np.random.seed(row['seed']);torch.cuda.synchronize();t=time.perf_counter()
    sp,tp,rel=matrices(azi_deg=row['pose'][0],ele_deg=row['pose'][1],radius=row['pose'][2],mean_depth=mean,fovy=2*np.arctan(24/38),device='cuda',dtype=torch.float16)
    if not all(torch.isfinite(v).all() for v in [sp,tp,rel]):raise ValueError('invalid camera')
    with torch.no_grad():result=nvs(src_image=source_half,src_depth=depth,rel_view_mtx=rel,src_proj_mtx=sp,tar_proj_mtx=tp)['synthesized'][0].float()
    if result.shape!=(3,512,512) or not torch.isfinite(result).all() or float(result.min())<0 or float(result.max())>1:raise ValueError('invalid generated tensor')
    torch.cuda.synchronize();seconds=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated();integrity=generator.commit_png(np.array(to_pil_image(result.cpu())),row['path'])
    saved={**row,'integrity':integrity,'run_id':run_id,'Slurm_job_id':job,'depth_sha256':depth_hash,'depth_dimensions':list(depth.shape),'source_dimensions':list(pil.size),'mean_depth':mean,'source_projection':sp.float().cpu().tolist(),'target_projection':tp.float().cpu().tolist(),'relative_view':rel.float().cpu().tolist(),'depth_seconds_once_per_source':depth_seconds,'generation_seconds':seconds,'peak_GPU_allocated_bytes':peak,'finite_tensor':True}
    
    if exp['condition']=='zero':
     resized=np.array(pil.resize((512,512),Image.Resampling.LANCZOS),dtype=np.float32);pixels=np.array(to_pil_image(result.cpu()),dtype=np.float32);saved['resynthesis_MAE_vs_resized_source']=float(np.abs(pixels-resized).mean())
     if saved['resynthesis_MAE_vs_resized_source']==0:raise ValueError('zero-motion copies original pixels')
    c.write(Path(row['path']).with_suffix('.json'),saved,exclusive=True);generated+=1
   print('SOURCE_COMPLETE',exp['key'],source['id'],'new_images',len(pending),'total',generated,flush=True)
 if a.pilot:c.write(c.QUERY/'evidence/pilot_generation.json',{'passed':all(x['exact_repeated_depth'] for x in depth_tests),'new_images':generated,'depth_reuse_tests':depth_tests,'generation_fingerprint':fingerprint,'Slurm_job_id':job})
 c.write(runpath/'completed.json',{'generated':generated,'seconds_including_loading':time.perf_counter()-started,'status':'completed'})
 print('GENERATION_COMPLETED',generated,flush=True)

