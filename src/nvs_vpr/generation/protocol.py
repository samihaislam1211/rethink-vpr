"""Metadata-only generation planning, integrity checks and numerical camera reference."""
import hashlib,json,math,os,re
from pathlib import Path
import numpy as np
from PIL import Image
from ..runtime import ROOT,SOURCE
GENERATION=ROOT/'generation';GEN=ROOT/'dependencies/genwarp'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def manifest(ds,role):
 cfg=read(ROOT/'baselines/config.json');d=next(x for x in cfg['datasets'] if x['name']==ds)
 return [json.loads(l) for l in (ROOT/'manifests'/d[role]).read_text().splitlines()]
def seed(index,view,base=42):
 if not(0<=index and 0<=view<100):raise ValueError('invalid seed coordinates')
 return base*1000000+index*100+view

def camera(az,el,radius,depth,focal=19.,sensor=24.,near=.01,far=100.):
 if depth<=0 or radius<0:raise ValueError('invalid scene/radius')
 a,e=np.deg2rad([az,el]);d=depth+radius
 target=np.array([-depth,0.,0.]);eye=np.array([d*np.cos(a)*np.cos(e),d*np.sin(a)*np.cos(e),d*np.sin(e)])+target
 def look(eye,at):
  z=eye-at;z/=np.linalg.norm(z);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
  out=np.eye(4);out[:3,:3]=np.stack([x,y,z]);out[:3,3]=-out[:3,:3]@eye;return out
 source=look(np.zeros(3),np.array([-1.,0.,0.]));tar=look(eye,target)
 fovy=2*np.arctan(sensor/(2*focal));proj=np.zeros((4,4));proj[0,0]=1/np.tan(fovy/2);proj[1,1]=-proj[0,0];proj[2,2]=-(far+near)/(far-near);proj[2,3]=-2*far*near/(far-near);proj[3,2]=-1
 return proj,tar@np.linalg.inv(source)
def verify_png(path,digest=None):
 p=Path(path)
 if not p.is_file():raise ValueError('missing synthetic image '+str(p))
 if digest and sha(p)!=digest:raise ValueError('synthetic image checksum mismatch')
 with Image.open(p) as im:
  if im.format!='PNG' or im.mode!='RGB' or im.size!=(512,512):raise ValueError('unexpected generated image encoding/dimensions')
  im.load()
 return {'sha256':sha(p),'bytes':p.stat().st_size,'width':512,'height':512,'format':'PNG','mode':'RGB'}
def plan(request):
 if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',request['run_id']):raise ValueError('unsafe run ID')
 schedules=read(GENERATION/'config/cameras.json')['schedules'];rows=[]
 if len(request['sources'])>2:raise ValueError('Generation execution/dry-run source cap is2')
 for source in request['sources']:
  role=source['role']
  if role not in ['reference','query']:raise ValueError('invalid source role')
  entries=manifest(source['dataset'],role);idx=source['index'];entry=entries[idx]
  if entry['index']!=idx or '/'+role+'/' not in entry['id']:raise ValueError('source-role/index mismatch')
  # Explicit IDs prevent accidentally selecting a different source after manifest drift.
  if source['id']!=entry['id']:raise ValueError('stable source ID mismatch')
  if len(source['views'])>8:raise ValueError('Generation pose cap8 per source')
  for choice in source['views']:
   s=schedules[choice['schedule']]
   if (source['dataset'],role) not in [tuple(p) for p in s['allowed']]:raise ValueError('schedule/dataset/role mismatch')
   vi=choice['view_index'];pose=s['poses'][vi]
   for repetition in range(choice.get('repetitions',1)):
    if choice.get('repetitions',1)>2:raise ValueError('repeat cap2')
    identity={'source_id':entry['id'],'source_sha256':sha(entry['path']),'schedule':choice['schedule'],'schedule_fingerprint':fp(s),'model_configuration_fingerprint':fp(read(GENERATION/'config/genwarp.json')),'run_id':request['run_id'],'view_index':vi,'repetition':repetition}
    uid=fp(identity)[:24];out=GENERATION/'runs'/request['run_id']/(uid+'.png')
    rows.append({'dataset':source['dataset'],'split':entry['id'].split('/')[1],'source_role':role,'source_id':entry['id'],'source_path':entry['path'],'source_index':idx,'source_sha256':sha(entry['path']),
      'view_index':vi,'azimuth':pose[0],'elevation':pose[1],'radius_offset':pose[2],'seed':seed(idx,vi),'depth_seed':seed(idx,0),
      'schedule':choice['schedule'],'schedule_fingerprint':fp(s),'preprocessing':choice.get('preprocessing_override',s['preprocessing']),'repetition':repetition,'synthetic_id':'synthetic/'+uid,'run_id':request['run_id'],'output_path':str(out)})
 if len(rows)>16:raise ValueError('Generation total image cap16 including repetitions')
 if len({x['output_path'] for x in rows})!=len(rows):raise ValueError('output identity collision')
 return rows

def resume_state(directory,rows,configuration_fingerprint):
 directory=Path(directory);record=directory/'run.json'
 if record.exists() and read(record)['configuration_fingerprint']!=configuration_fingerprint:raise ValueError('run configuration mismatch')
 stored=[]
 if (directory/'manifest.jsonl').exists():stored=[json.loads(x) for x in (directory/'manifest.jsonl').read_text().splitlines()]
 byid={x['synthetic_id']:x for x in stored}
 if len(byid)!=len(stored):raise ValueError('duplicate generated manifest identities')
 planned={r['synthetic_id']:r for r in rows}
 if set(byid)-set(planned):raise ValueError('unplanned stored view')
 todo=[]
 for row in rows:
  p=Path(row['output_path']);saved=byid.get(row['synthetic_id'])
  if saved:
   if any(saved[k]!=v for k,v in row.items()):raise ValueError('stored/planned provenance mismatch')
   verify_png(p,saved['integrity']['sha256'])
  elif p.exists():raise ValueError('untracked output collision; use a new run ID')
  else:todo.append(row)
 return stored,todo
