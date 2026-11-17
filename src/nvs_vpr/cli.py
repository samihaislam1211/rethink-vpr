"""Configure a run and submit bounded scheduler-allocated VPR workloads."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
COUNTS={'GardensPoint':(200,200),'StLucia':(200,200),'Pittsburgh30k':(10000,6816),'Tokyo247':(75984,315)}
def read(p):return json.loads(Path(p).read_text())

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def absolute(value):
    p=Path(value).expanduser()
    if not p.is_absolute() or '\n' in str(p) or "'" in str(p):raise ValueError('Use absolute paths without newline/single-quote characters')
    return p.resolve()

def validate(cfg,assets=False):
    for k in ['work_dir','vpr_python','genwarp_python','descriptor_checkpoints','genwarp_checkpoints','zoe_source','midas_source','depth_checkpoint']:absolute(cfg[k])
    if not 1<=int(cfg['slurm']['max_gpus'])<=6:raise ValueError('max_gpus must be 1..6')
    if set(cfg['datasets'])!=set(COUNTS):raise ValueError('Configure all four supported datasets')
    missing=[]
    for ds,d in cfg['datasets'].items():
        for role in ['reference','query']+(['gt_file'] if ds=='StLucia' else []):
            p=absolute(d[role])
            if assets and not p.exists():missing.append(str(p))
    if assets:
        for k in ['vpr_python','genwarp_python','descriptor_checkpoints','genwarp_checkpoints','zoe_source','midas_source','depth_checkpoint']:
            if not absolute(cfg[k]).exists():missing.append(cfg[k])
        for name in ['ResNet50_2048_cosplace.pth','ResNet50_2048_eigenplaces.pth','alexnet-owt-7be5be79.pth','resnet50-11ad3fa6.pth']:
            if not (absolute(cfg['descriptor_checkpoints'])/name).is_file():missing.append(str(absolute(cfg['descriptor_checkpoints'])/name))
    if assets:
        for rel in ['multi1/config.json','multi1/denoising_unet.pth','multi1/pose_guider.pth','multi1/reference_unet.pth','image_encoder/config.json','image_encoder/pytorch_model.bin','sd-vae-ft-mse/config.json','sd-vae-ft-mse/diffusion_pytorch_model.safetensors']:
            candidate=absolute(cfg['genwarp_checkpoints'])/rel
            if not candidate.is_file():missing.append(str(candidate))
    return missing

def prepare(cfg):
    validate(cfg);work=absolute(cfg['work_dir'])
    if work.exists():raise FileExistsError('work_dir must be new; existing runs are never overwritten')
    if work==REPO or REPO.is_relative_to(work):raise ValueError('Choose a dedicated output directory')
    work.mkdir(parents=True)
    for component in ['baselines','generation','database','query','analysis']:
        for folder in ['config','cache','logs','runs','results','evidence','manifests']:(work/component/folder).mkdir(parents=True,exist_ok=True)
    for folder in ['config','manifests','logs','jobs','stage_status']:(work/folder).mkdir(exist_ok=True)
    dependencies=work/'dependencies/genwarp';dependencies.mkdir(parents=True)
    source=REPO/'integrations/genwarp_local'
    for name in ['genwarp','extract_latents.py']:(dependencies/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
    (dependencies/'checkpoints').symlink_to(absolute(cfg['genwarp_checkpoints']),target_is_directory=True)
    (dependencies/'extern').mkdir();(dependencies/'extern/ZoeDepth').symlink_to(absolute(cfg['zoe_source']),target_is_directory=True)
    datasets=[]
    for name,entry in cfg['datasets'].items():
        split='original' if name in ['GardensPoint','StLucia'] else 'test'
        item={'name':name,'reference':f'{name}__{split}__reference.jsonl','query':f'{name}__{split}__query.jsonl'}
        if name=='StLucia':item['gt_file']=str(absolute(entry['gt_file']))
        datasets.append(item)
    save(work/'baselines/config.json',{'datasets':datasets,'models':['CosPlace','EigenPlaces','AlexNet-GeM','ResNet50-GeM']})
    model=read(REPO/'configs/genwarp.json');model.update(genwarp_root=str(dependencies),checkpoint_root=str(absolute(cfg['genwarp_checkpoints'])),depth_checkpoint=str(absolute(cfg['depth_checkpoint'])),zoe_source=str(absolute(cfg['zoe_source'])),midas_source=str(absolute(cfg['midas_source'])))
    save(work/'generation/config/genwarp.json',model)
    shutil.copyfile(REPO/'configs/cameras.json',work/'generation/config/cameras.json')
    shutil.copyfile(REPO/'configs/database.json',work/'database/config/protocol.json')
    save(work/'config/project.json',cfg)
    sources=list((REPO/'src').rglob('*.py'))+list((REPO/'configs').rglob('*.json'))+[REPO/'run.py']
    save(work/'SOURCE_SHA256.json',{str(p.relative_to(REPO)):sha(p) for p in sources})
    print('Run configured at',work,'\nNext: run the smoke workflow. No jobs have been submitted.')

GPU={'baseline','baseline-smoke','genwarp-smoke','database-pilot-generate','database-pilot-features','database-generate','database-features','query-pilot-generate','query-pilot-features','query-generate','query-features','robust-features'}
STAGES=sorted(GPU|{'prepare-data','audit-models','database-prepare','database-pilot-evaluate','database-evaluate','query-prepare','query-null','query-merge','query-evaluate','robust-prepare','robust-merge','robust-evaluate','gate-signals','gating','statistics','plots'})

def submit(cfg,args):
    if args.stage not in STAGES:raise ValueError('unknown stage')
    work=absolute(cfg['work_dir']);local=read(work/'config/project.json')
    if local!=cfg:raise ValueError('configuration changed after setup; create a new run')
    for relative,digest in read(work/'SOURCE_SHA256.json').items():
        if sha(REPO/relative)!=digest:raise ValueError('application source/configuration changed after setup: '+relative)
    logs=work/'logs';logs.mkdir(exist_ok=True);jobs=work/'jobs';jobs.mkdir(exist_ok=True)
    # Read scheduler state before any allocation; these queries do not mutate other jobs.
    inspected={}
    for cmd in [['squeue','-u',str(os.getuid())],['sinfo','-N','-l'],['scontrol','show','nodes'],['scontrol','show','partition',cfg['slurm']['partition']]]:
        r=subprocess.run(cmd,capture_output=True,text=True);inspected[' '.join(cmd)]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        if r.returncode:raise RuntimeError(r.stderr)
    save(work/'scheduler_before_submission.json',inspected)
    if args.array:
        import re
        if not re.fullmatch(r'0-[1-9][0-9]*',args.array):raise ValueError('--array format must be 0-N, N>=1')
    if args.index<0 or args.shards<1:raise ValueError('index must be nonnegative and shards positive')
    limits={'baseline':15,'database-features':4,'database-evaluate':4,'robust-features':17,'robust-merge':2,'robust-evaluate':7,'gating':7}
    for stage in ['database-generate','query-generate','query-null','query-features']:limits[stage]=args.shards-1
    maximum=limits.get(args.stage,0)
    if args.index>maximum or (args.array and int(args.array.split('-')[1])>maximum):raise ValueError('index/array exceeds valid range for this stage')
    gpu=args.stage in GPU;python=cfg['genwarp_python'] if 'generate' in args.stage or args.stage=='genwarp-smoke' else cfg['vpr_python']
    command=[python,'-B',str(REPO/'run.py'),'--config',str(work/'config/project.json'),'execute',args.stage,'--index',str(args.index),'--shards',str(args.shards)]
    shell='#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport OMP_NUM_THREADS='+str(cfg['slurm']['cpus'])+'\nexport OPENBLAS_NUM_THREADS='+str(cfg['slurm']['cpus'])+'\nexport MAX_JOBS='+str(cfg['slurm']['cpus'])+'\nexport MPLCONFIGDIR='+shlex.quote(str(work/'mplcache'))+'\nexport TORCH_EXTENSIONS_DIR='+shlex.quote(str(work/'torch_extensions'))+'\n'
    if gpu and ('generate' in args.stage or args.stage=='genwarp-smoke'):shell+='export CUDA_HOME='+shlex.quote(str(cfg.get('cuda_home',Path(python).parent.parent)))+'\n'
    shell+='cd '+shlex.quote(str(REPO))+'\nexec '+(shlex.join(command[:-4])+' --index "${SLURM_ARRAY_TASK_ID}" --shards '+str(args.shards) if args.array else shlex.join(command))+'\n';script=jobs/(args.stage+'_'+(args.array or str(args.index))+'.sh');script.write_text(shell)
    s=cfg['slurm'];cmd=['sbatch','--parsable','--partition='+s['partition'],'--nodes=1','--ntasks=1','--cpus-per-task='+str(s['cpus']),'--mem='+s['memory'],'--time='+s['time'],'--job-name=vpr_'+args.stage,'--output='+str(logs/(args.stage+'_%j.out'))]
    if gpu:cmd+=['--gres='+s['gpu_gres']]
    if args.after:
        if not all(x.isdigit() for x in args.after.split(':')):raise ValueError('--after must be job IDs separated by colon')
        cmd+=['--dependency=afterok:'+args.after,'--kill-on-invalid-dep=yes']
    cmd+=[str(script)]
    journal=work/'SUBMISSIONS.json';records=read(journal) if journal.exists() else []
    active=[]
    state=subprocess.run(['squeue','-h','-u',str(os.getuid()),'-o','%i|%j|%b|%t'],capture_output=True,text=True,check=True).stdout
    if gpu:
        # Include all this user's allocated/pending GPU requests, not just this project's names.
        import re
        for line in state.splitlines():
            f=line.split('|')
            if len(f)>2 and 'gpu' in f[2].lower():active.append(sum(int(n) for n in re.findall(r':(\d+)(?:\(|,|$)',f[2])))
        if sum(active)>=int(s['max_gpus']):raise RuntimeError('Configured GPU concurrency already reached; wait for a job to finish')
    if args.array:cmd.insert(-1,'--array='+args.array+'%'+str(max(1,int(s['max_gpus'])-sum(active)) if gpu else int(s['max_gpus'])))
    r=subprocess.run(cmd,capture_output=True,text=True);record={'stage':args.stage,'index':args.index,'command':cmd,'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr};records.append(record);save(journal,records)
    if r.returncode:raise RuntimeError(r.stderr)
    print(r.stdout.strip())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True)
    commands=ap.add_subparsers(dest='action',required=True)
    commands.add_parser('setup');commands.add_parser('check')
    for name in ['submit','execute']:
        command=commands.add_parser(name);command.add_argument('stage',choices=STAGES);command.add_argument('--index',type=int,default=0);command.add_argument('--shards',type=int,default=6)
        if name=='submit':command.add_argument('--after');command.add_argument('--array',help='0-N, with bounded concurrent tasks')
    args=ap.parse_args();cfg=read(args.config)
    if args.action=='setup':prepare(cfg)
    elif args.action=='check':
        missing=validate(cfg,True);print(json.dumps({'ready':not missing,'missing_assets':missing},indent=2));sys.exit(bool(missing))
    elif args.action=='submit':submit(cfg,args)
    else:
        if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Use submit or run inside a real Slurm allocation')
        from . import pipeline
        pipeline.initialize(cfg['work_dir']);pipeline.run_stage(args.stage,args.index,args.shards)

