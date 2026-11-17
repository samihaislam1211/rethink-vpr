"""Local-only GenWarp adapter. Dry-run is CPU/metadata-only. Generation hard caps apply."""
import argparse,contextlib,fcntl,importlib.util,json,os,random,socket,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
from PIL import Image
from . import protocol as c

def local_models(cfg):
 import torch
 sys.path[:0]=[cfg['genwarp_root'],cfg['zoe_source'],cfg['midas_source']]
 # The original full ZoeD_N checkpoint includes its MiDaS weights. Construct that architecture locally.
 from midas.dpt_depth import DPTDepthModel
 from zoedepth.models.base_models.midas import MidasCore
 from zoedepth.utils.config import get_config
 from zoedepth.models.builder import build_model
 def offline_build(midas_model_type='DPT_BEiT_L_384',train_midas=False,use_pretrained_midas=False,fetch_features=False,freeze_bn=True,force_keep_ar=False,force_reload=False,**kwargs):
  if midas_model_type!='DPT_BEiT_L_384' or use_pretrained_midas or force_reload:raise ValueError('Unsupported offline MiDaS request')
  if 'img_size' in kwargs:kwargs=MidasCore.parse_img_size(kwargs)
  size=kwargs.pop('img_size',[384,384]);kwargs.update(keep_aspect_ratio=force_keep_ar)
  midas=DPTDepthModel(path=None,backbone='beitl16_384',non_negative=True)
  core=MidasCore(midas,trainable=train_midas,fetch_features=fetch_features,freeze_bn=freeze_bn,img_size=size,**kwargs)
  core.set_output_channels(midas_model_type);return core
 original=MidasCore.build;MidasCore.build=staticmethod(offline_build)
 try:
  zcfg=get_config('zoedepth','infer',pretrained_resource='local::'+cfg['depth_checkpoint'],use_pretrained_midas=False)
  depth=build_model(zcfg).to('cuda').eval().requires_grad_(False)
 finally:MidasCore.build=original
 from genwarp import GenWarp
 from extract_latents import make_camera_matrices,load_and_crop
 nvs=GenWarp(cfg={k:cfg[k] for k in ['checkpoint_name','height','width','num_inference_steps','guidance_scale','half_precision_weights']}|{'pretrained_model_path':cfg['checkpoint_root']},device='cuda')
 for name in ['vae','image_encoder','reference_unet','denoising_unet','pose_guider']:getattr(nvs,name).eval()
 return nvs,depth,make_camera_matrices,load_and_crop

def deny_download(*args,**kwargs):raise RuntimeError('Network/Torch Hub downloads disabled for Generation local-only inference')
def commit_png(array,path):
 path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  Image.fromarray(array).save(tmp,format='PNG')
  os.link(tmp,path)  # atomic no-clobber installation
 finally:
  if tmp.exists():tmp.unlink()
 return c.verify_png(path)
def commit_manifest(path,rows):
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));os.replace(tmp,path)

def run(request,execute=False,resume=False):
 rows=c.plan(request);cfg=c.read(c.GENERATION/'config/genwarp.json');originals=c.read(c.GENERATION/'evidence/genwarp_originals.json')
 config_fp=c.fp({'request':request,'rows':rows,'model':cfg,'checkpoint_and_source_hashes':originals,'adapter_sha256':c.sha(__file__)})
 directory=c.GENERATION/'runs'/request['run_id']
 if not execute:return {'dry_run':True,'configuration_fingerprint':config_fp,'model':cfg,'rows':rows}
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU inference requires a Slurm allocation')
 if directory.exists() and not resume:raise FileExistsError('Use explicit --resume or a new run ID')
 stored,todo=c.resume_state(directory,rows,config_fp)
 if not todo:return {'status':'resume_complete','records':len(stored),'configuration_fingerprint':config_fp,'models_loaded':False}
 for path,snapshot in originals.items():
  stat=Path(path).stat()
  if stat.st_size!=snapshot['bytes'] or stat.st_mtime_ns!=snapshot['mtime_ns']:raise ValueError('Source/checkpoint changed since preflight: '+path)
  if Path(path).suffix=='.py' and c.sha(path)!=snapshot['sha256']:raise ValueError('Generation code changed')
 directory.mkdir(parents=True,exist_ok=True)
 lease=(directory/'.generation.lock').open('a')
 fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 stored,todo=c.resume_state(directory,rows,config_fp)
 if not todo:return {'status':'resume_complete','records':len(stored),'models_loaded':False}
 import torch,importlib.metadata as md
 if not torch.cuda.is_available():raise RuntimeError('Allocated node has no CUDA')
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.hub.load=deny_download;torch.hub.load_state_dict_from_url=deny_download;torch.hub.download_url_to_file=deny_download
 socket.socket.connect=deny_download
 env={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'packages':{p:md.version(p) for p in ['torchvision','diffusers','transformers','huggingface_hub','timm','numpy','Pillow','splatting']},'gpu':torch.cuda.get_device_name(0),'job_id':os.environ['SLURM_JOB_ID']}
 c.write(directory/'run.json',{'configuration_fingerprint':config_fp,'request':request,'model':cfg,'originals':originals,'environment':env,'adapter_sha256':c.sha(__file__),'common_sha256':c.sha(c.SOURCE/'generation/protocol.py'),'network_disabled':True})
 start=time.perf_counter();torch.manual_seed(42)
 nvs,zoe,matrices,crop=local_models(cfg);load_seconds=time.perf_counter()-start
 c.write(directory/'loaded_models.json',{'scheduler_class':type(nvs.scheduler).__name__,'scheduler_config':dict(nvs.scheduler.config),'timesteps':nvs.scheduler.timesteps.cpu().tolist(),'GenWarp_config':dict(nvs.cfg),'depth_model_class':type(zoe).__name__,'depth_eval':not zoe.training,'model_modes':{name:not getattr(nvs,name).training for name in ['vae','image_encoder','reference_unet','denoising_unet','pose_guider']},'model_loading_seconds':load_seconds})
 from torchvision.transforms.functional import to_tensor,to_pil_image
 for source_id in dict.fromkeys(row['source_id'] for row in todo):
  source_rows=[r for r in todo if r['source_id']==source_id]
  for row in source_rows:
   # Re-estimate depth for the repeated configuration to test complete inference reproducibility.
   torch.manual_seed(row['depth_seed']);random.seed(row['depth_seed']);np.random.seed(row['depth_seed'])
   policy=row['preprocessing']
   if policy=='center_square_lanczos512':pil=crop(row['source_path'],512)
   elif policy=='raw_rgb':
    with Image.open(row['source_path']) as im:pil=im.convert('RGB')
   else:raise ValueError('Unresolved preprocessing policy '+policy)
   srgeneration_protocol2=to_tensor(pil)[None].cuda().float();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
   with torch.no_grad():depth=zoe.infer(srgeneration_protocol2,pad_input=True,with_flip_aug=True).half()
   if not torch.isfinite(depth).all() or (depth<=0).any():raise ValueError('Invalid depth tensor')
   mean=float(depth.mean().cpu());source=srgeneration_protocol2.half()
   torch.manual_seed(row['seed']);random.seed(row['seed']);np.random.seed(row['seed'])
   fovy=2*np.arctan(cfg['sensor_width_mm']/cfg['focal_mm']/2)
   sp,tp,rel=matrices(azi_deg=row['azimuth'],ele_deg=row['elevation'],radius=row['radius_offset'],mean_depth=mean,fovy=fovy,device='cuda',dtype=torch.float16)
   for tensor in [sp,tp,rel]:
    if not torch.isfinite(tensor).all():raise ValueError('Invalid camera matrix')
   with torch.no_grad():renders=nvs(src_image=source,src_depth=depth,rel_view_mtx=rel,src_proj_mtx=sp,tar_proj_mtx=tp)
   output=renders['synthesized'][0].float()
   if not torch.isfinite(output).all() or tuple(output.shape)!=(3,512,512):raise ValueError('Invalid synthesized tensor')
   if float(output.min())<0 or float(output.max())>1:raise ValueError('Unexpected synthesized range')
   torch.cuda.synchronize();seconds=time.perf_counter()-t0;memory=torch.cuda.max_memory_allocated()
   array=np.array(to_pil_image(output.cpu()));integrity=commit_png(array,row['output_path'])
   record={**row,'checkpoint_identifiers':{path:v['sha256'] for path,v in originals.items() if not path.endswith('.py')},'generation_parameters':cfg,'configuration_fingerprint':config_fp,'integrity':integrity,'mean_depth':mean,'depth_dimensions':list(depth.shape),'actual_source_dimensions':list(pil.size),'source_projection':sp.float().cpu().tolist(),'target_projection':tp.float().cpu().tolist(),'relative_view':rel.float().cpu().tolist(),'runtime_seconds_including_depth':seconds,'peak_GPU_allocated_bytes':memory,'finite_tensor':True,'output_range':[float(output.min()),float(output.max())]}
   stored.append(record);commit_manifest(directory/'manifest.jsonl',stored)
   print('GENERATED',record['synthetic_id'],record['integrity']['sha256'],'seconds',seconds,'peak_bytes',memory,flush=True)
 comparisons=[]
 for source_id in {r['source_id'] for r in stored}:
  pair=[r for r in stored if r['source_id']==source_id and r['view_index']==0 and r['schedule']!='near_zero_v1']
  if len(pair)==2:
   a,b=[np.array(Image.open(r['output_path'])).astype(np.float64) for r in pair];delta=a-b;mse=float(np.mean(delta**2))
   comparisons.append({'source_id':source_id,'png_byte_identical':pair[0]['integrity']['sha256']==pair[1]['integrity']['sha256'],'pixel_MAE':float(np.abs(delta).mean()),'pixel_max_abs':float(np.abs(delta).max()),'pixel_RMSE':float(np.sqrt(mse)),'PSNR_dB':None if mse==0 else float(10*np.log10(255**2/mse)),'configurations_identical_except_repeat_identity':all(pair[0][k]==pair[1][k] for k in ['source_sha256','preprocessing','azimuth','elevation','radius_offset','seed','depth_seed','mean_depth'])})
 c.write(directory/'reproducibility.json',comparisons)
 return {'status':'complete','new_records':len(stored),'comparisons':comparisons,'environment':env}

