import argparse,json,os,platform,sys,time
from pathlib import Path
import numpy as np
from . import data as c

def model_spec(name):
 files={'CosPlace':'ResNet50_2048_cosplace.pth','EigenPlaces':'ResNet50_2048_eigenplaces.pth','AlexNet-GeM':'alexnet-owt-7be5be79.pth','ResNet50-GeM':'resnet50-11ad3fa6.pth'}
 p=c.ROOT/'config/project.json';p=Path(json.loads(p.read_text())['descriptor_checkpoints'])/files[name]
 if not p.is_file():raise FileNotFoundError(p)
 return {'name':name,'checkpoint':str(p),'sha256':c.sha(p),'dim':256 if name=='AlexNet-GeM' else 2048,'architecture':('AlexNet conv+GeM p3' if name=='AlexNet-GeM' else 'ResNet50 conv+GeM p3' if name=='ResNet50-GeM' else 'ResNet50 conv+channelL2+learnedGeM+FC2048+L2'),'pool_eps':1e-6}
def build(spec):
 import torch,torchvision
 from torch import nn
 import torch.nn.functional as F
 class L2(nn.Module):
  def forward(self,x):return F.normalize(x,p=2,dim=1)
 class GeM(nn.Module):
  def __init__(self,learned):
   super().__init__();self.p=nn.Parameter(torch.ones(1)*3) if learned else 3.
  def forward(self,x):return F.avg_pool2d(x.clamp(min=1e-6).pow(self.p),x.shape[-2:]).pow(1/self.p)
 class Net(nn.Module):
  def __init__(self):
   super().__init__();base=torchvision.models.resnet50(weights=None);self.backbone=nn.Sequential(*list(base.children())[:-2]);self.aggregation=nn.Sequential(L2(),GeM(True),nn.Flatten(),nn.Linear(2048,2048),L2())
  def forward(self,x):return self.aggregation(self.backbone(x))
 state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
 if spec['name'] in ['CosPlace','EigenPlaces']:
  model=Net();model.load_state_dict(state,strict=True)
 else:
  base=torchvision.models.alexnet(weights=None) if spec['name']=='AlexNet-GeM' else torchvision.models.resnet50(weights=None)
  base.load_state_dict(state,strict=True)
  backbone=base.features if spec['name']=='AlexNet-GeM' else nn.Sequential(*list(base.children())[:-2])
  model=nn.Sequential(backbone,GeM(False),nn.Flatten())
 model.eval().requires_grad_(False);return model.cuda()
def extract(ds,role,spec,model,env,limit=None):
 import torch,torchvision
 from PIL import Image
 data=c.manifest(ds,role);data=data[:limit] if limit else data
 transform=torchvision.transforms.Compose([torchvision.transforms.Resize(480,interpolation=torchvision.transforms.InterpolationMode.BILINEAR,antialias=True),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 meta=c.cache_meta(ds,role,spec,env) if not limit else {};p=c.ROOT/'baselines/cache'/f"{ds['name']}__{spec['name']}__{role}.npy"
 if not limit and p.exists():return c.load_cache(p,meta)
 # Batch by equal shape; bounding memory while preserving manifest row order.
 out=np.empty((len(data),spec['dim']),dtype=np.float32);batch=[];ids=[]
 def flush():
  if not batch:return
  with torch.inference_mode():f=torch.nn.functional.normalize(model(torch.stack(batch).cuda()).float(),dim=1).cpu().numpy()
  if f.shape!=(len(batch),spec['dim']) or not np.isfinite(f).all():raise ValueError('invalid features')
  out[ids]=f;batch.clear();ids.clear()
 for i,row in enumerate(data):
  with Image.open(row['path']) as im:t=transform(im.convert('RGB'))
  if batch and (len(batch)>=8 or t.shape!=batch[0].shape):flush()
  batch.append(t);ids.append(i)
  if i%1000==0:print(ds['name'],spec['name'],role,i,'/',len(data),flush=True)
 flush()
 if not np.allclose(np.linalg.norm(out,axis=1),1,atol=1e-5):raise ValueError('normalization failed')
 if not limit:
  tmp=p.with_suffix('.tmp.npy');np.save(tmp,out);tmp.replace(p)
  c.write(p.with_suffix('.json'),{'fingerprint':c.fingerprint(meta),'provenance':meta,'embedding_sha256':c.sha(p),'shape':list(out.shape)})
 return out


def main():
 import torch,torchvision
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--model');ap.add_argument('--dataset');a=ap.parse_args()
 if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires allocated Slurm CUDA node')
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 env={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'cuda':torch.version.cuda}
 execution={'environment':env,'job_id':os.getenv('SLURM_JOB_ID'),'hostname':platform.node(),'gpu':torch.cuda.get_device_name(),'command':sys.argv,'seed':42}
 cfg=json.loads((c.ROOT/'baselines/config.json').read_text());models=[a.model] if a.model else cfg['models']
 for name in models:
  spec=model_spec(name);m=build(spec);spec['pool_p']=float(m.aggregation[1].p.detach().cpu()) if name in ['CosPlace','EigenPlaces'] else 3.0;print('LOADED',name,spec,flush=True)
  for ds in cfg['datasets']:
   if a.dataset and ds['name']!=a.dataset:continue
   started=time.time();r=extract(ds,'reference',spec,m,env,2 if a.smoke else None);q=extract(ds,'query',spec,m,env,2 if a.smoke else None)
   ix,score=c.topk(r,q)
   if a.smoke:continue
   gt=json.loads((c.ROOT/'baselines'/f"{ds['name']}_gt.json").read_text());metrics={k:c.metrics(ix,g) for k,g in gt.items()}
   rr,qq=c.manifest(ds,'reference'),c.manifest(ds,'query');stem=f"{ds['name']}__{name}"
   with (c.ROOT/'baselines/results'/f'{stem}_predictions.jsonl').open('w') as f:
    for j,row in enumerate(qq):f.write(json.dumps({'query_id':row['id'],'reference_ids':[rr[i]['id'] for i in ix[j]],'reference_indices':ix[j].tolist(),'scores':score[j].tolist(),'correct':{p:{k:v['flags'][int(k)][j] for k in ['1','5','10']} for p,v in metrics.items()}})+'\n')
   for metric in metrics.values():metric.pop('flags')
   c.write(c.ROOT/'baselines/results'/f'{stem}.json',{'dataset':ds['name'],'model':name,'reference_count':len(r),'query_count':len(q),'metrics':metrics,'execution':execution,'model_spec':spec,'seconds':time.time()-started,'manifest_hashes':{role:c.sha(c.ROOT/'manifests'/ds[role]) for role in ['reference','query']},'GT_sha256':c.sha(c.ROOT/'baselines'/f"{ds['name']}_gt.json"),'tie_policy':'score descending then original reference index ascending'})
  del m;torch.cuda.empty_cache()
 c.write(c.ROOT/'baselines/logs'/f"execution_{os.getenv('SLURM_JOB_ID')}.json",{**execution,'status':'success','smoke':a.smoke})
