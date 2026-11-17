#!/usr/bin/env python3
"""Run dataset preparation, synthesis, retrieval and statistical analysis."""
import argparse,json,os,shutil,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[2]

def initialize(work):
    global W,P,CFG,C
    W=Path(work).resolve();P=W;os.environ['NVS_VPR_WORK_DIR']=str(W)
    CFG=json.loads((W/'config/project.json').read_text())
    from . import runtime
    runtime.ROOT=W
    from .database import common
    for name,module in list(sys.modules.items()):
        if name.startswith('nvs_vpr.') and hasattr(module,'ROOT'):module.ROOT=W
        for attr,folder in [('GENERATION','generation'),('DATABASE','database'),('QUERY','query'),('ANALYSIS','analysis')]:
            if name.startswith('nvs_vpr.') and hasattr(module,attr):setattr(module,attr,W/folder)
    from .generation import protocol
    protocol.GEN=W/'dependencies/genwarp'
    C=common

def prepare_data():
    from .cli import COUNTS
    rows=[]
    for d in C.read(P/'baselines/config.json')['datasets']:
        name=d['name'];spec=CFG['datasets'][name];split='original' if name in ['GardensPoint','StLucia'] else 'test'
        for role,n in zip(['reference','query'],COUNTS[name]):
            directory=Path(spec[role]);files=sorted(directory.glob('*.jpg'))
            if len(files)!=n:raise ValueError(f'{name}/{role}: expected {n}, found {len(files)}; use the documented exact subset/format')
            values=[{'index':i,'id':f'{name}/{split}/{role}/{f.name}','filename':f.name,'path':str(f.resolve())} for i,f in enumerate(files)]
            path=P/'manifests'/d[role];path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in values));rows.append({'dataset':name,'role':role,'count':n,'manifest_sha256':C.sha(path)})
        gt=C.base.positives(d)
        if name=='StLucia' and [sum(bool(x) for x in gt[g]) for g in ['hard','soft']]!=[165,200]:raise ValueError('Use original StLucia_small GT.npz: hard165/soft200, not regenerated soft labels')
        C.write(P/'baselines'/f'{name}_gt.json',gt)
    C.write(W/'data_preparation.json',{'status':'pass','rows':rows,'dataset_bytes_modified':False})

def audit_models():
    model=C.read(P/'generation/config/genwarp.json');paths=[]
    for group in ['multi1','image_encoder','sd-vae-ft-mse']:
        root=Path(model['checkpoint_root'])/group
        if not root.is_dir():raise FileNotFoundError(root)
        paths.extend(p for p in root.rglob('*') if p.is_file())
    paths.append(Path(model['depth_checkpoint']))
    paths.extend((REPO/'integrations/genwarp_local/genwarp').rglob('*.py'))
    paths.append(REPO/'integrations/genwarp_local/extract_latents.py')
    paths.extend([Path(model['zoe_source'])/'zoedepth/models/base_models/midas.py',Path(model['midas_source'])/'midas/dpt_depth.py'])
    record={}
    for p in paths:
        s=p.stat();record[str(p)]={'sha256':C.sha(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    C.write(P/'generation/evidence/genwarp_originals.json',record)
    C.write(W/'model_audit.json',{'status':'pass','files':len(record),'generation_checkpoint':'multi1','full_local_SHA256_recorded':True})

def csr(ds,directory):
    import numpy as np
    gt=C.read(P/'baselines'/f'{ds}_gt.json');arrays={}
    for name,lists in gt.items():
        arrays[name+'_ptr']=np.r_[0,np.cumsum([len(a) for a in lists])].astype(np.int64)
        arrays[name+'_indices']=np.array([i for a in lists for i in a],np.int32)
    C.write_jsonl(directory/f'real__{ds}__reference.jsonl.gz',C.real(ds,'reference'))
    C.write_jsonl(directory/f'real__{ds}__query.jsonl.gz',C.real(ds,'query'))
    np.savez_compressed(directory/f'GT__{ds}.npz',**arrays)

def database_prepare():
    import numpy as np
    protocol=C.read(P/'database/config/protocol.json');sources=C.read(REPO/'configs/selection/database_sources.json')
    for exp in protocol['experiments']:
        selected=[];refs=C.real(exp['dataset'],'reference')
        for item in sources[exp['key']]['sources']:
            r=refs[item['index']]
            if r['id']!=item['id']:raise ValueError('Database source IDs differ; match public dataset naming/order before rerunning')
            selected.append(dict(r,sha256=C.sha(r['path'])))
        if len(selected)!=exp['sources']:raise ValueError('selected source count mismatch')
        C.write(P/'database/manifests'/f"sources__{exp['key']}.json",{'dataset':exp['dataset'],'source_role':'reference','sources':selected,'selection':'preserved exact source indices and IDs; source SHA recomputed from supplied images'})
        if exp['dataset']=='Pittsburgh30k':shutil.copy2(REPO/'configs/selection'/f"cohort__{exp['key']}.json",P/'database/config'/f"cohort__{exp['key']}.json")
    checks=[]
    for ds in C.read(P/'baselines/config.json')['datasets']:
        name=ds['name'];csr(name,P/'database/manifests');original=C.jsonl(P/'baselines/results'/f'{name}__CosPlace_predictions.jsonl');C.write_jsonl(P/'database/evidence'/f'baselines_baseline__{name}.jsonl.gz',original);C.write(P/'database/evidence'/f'baselines_result__{name}.json',C.read(P/'baselines/results'/f'{name}__CosPlace.json'))
        start=time.perf_counter();db=C.load_real(name,'reference');queries=C.load_real(name,'query');ix,sc=C.base.topk(db,queries)
        if ix.tolist()!=[r['reference_indices'] for r in original] or sc.tolist()!=[r['scores'] for r in original]:raise ValueError('Independent real-only B0 reranking differs; stop and investigate')
        C.write_jsonl(P/'database/evidence'/f'B0_recomputed__{name}.jsonl.gz',[{'query_index':i,'query_id':r['query_id'],'reference_indices':ix[i].tolist(),'scores':sc[i].tolist()} for i,r in enumerate(original)])
        checks.append({'dataset':name,'queries':len(queries),'rank_identical':True,'scores_byte_identical':True,'retrieval_seconds':time.perf_counter()-start})
    C.write(P/'database/evidence/B0_equivalence.json',{'passed':True,'datasets':checks,'scope':'fresh full real-only rank and score reranking; not historical replay'})
    C.write(P/'database/evidence/preflight.json',{'status':'pass','scope':'fresh selected-source identities, original real manifests/GT and B0; no arm A available'})

def query_prepare():
    from .query import common as query_config
    baseline=C.jsonl(P/'baselines/results/Pittsburgh30k__CosPlace_predictions.jsonl');indices=[i for i,r in enumerate(baseline) if not r['correct']['radius50m']['1']]
    if not indices:raise ValueError('No baseline failures at50m; near-identity recovery control has no selected population')
    C.write(P/'query/manifests/selected_failures.json',{'indices':indices,'query_ids':[baseline[i]['query_id'] for i in indices],'selection':'original query rank1 failure under50m GT; recovery separately25m/50m'})
    for ds in query_config.DATASETS:
        refs=C.real(ds,'reference');queries=C.real(ds,'query');C.write(P/'query/manifests'/f'sources__{ds}.json',{'sources':[dict(r,sha256=C.sha(r['path'])) for r in queries]});csr(ds,P/'query/manifests');C.write_jsonl(P/'query/evidence'/f'baseline__{ds}.jsonl.gz',C.jsonl(P/'baselines/results'/f'{ds}__CosPlace_predictions.jsonl'))
    C.write(P/'query/evidence/preflight.json',{'status':'pass','scope':'fresh generation only; original identities/counts, full source hashes,25/50m GT and computed50m failure selection; no historical-asset claims'})

def robust_prepare():
    from .analysis import common as analysis_config
    for ds in analysis_config.DATASETS:
        csr(ds,P/'analysis/manifests')
        for model in analysis_config.MODELS:C.write_jsonl(P/'analysis/evidence'/f'baseline__{ds}__{model}.jsonl.gz',C.jsonl(P/'baselines/results'/f'{ds}__{model}_predictions.jsonl'))
        for condition in analysis_config.CONDITIONS:
            key=f'fresh__{ds}__{condition}';C.write_jsonl(P/'analysis/manifests'/f'{key}.jsonl.gz',C.jsonl(P/'query/manifests'/f'{key}.jsonl.gz'));C.atomic_bytes(P/'analysis/evidence/query_predictions'/f'{key}__predictions.jsonl.gz',(P/'query/results'/f'{key}__predictions.jsonl.gz').read_bytes())

def summary():
    import csv,io
    rows=[]
    for path in sorted((P/'analysis/results').glob('summary__fresh__*.json')):
        for label,r in C.read(path).items():rows.append([r['arm'],r['dataset'],r['model'],r['condition'],r['fusion'],r['GT'],r['n'],r['baseline_recall']['1'],r['expanded_recall']['1'],r['paired']['gains'],r['paired']['losses'],r['paired']['delta']])
    buf=io.StringIO();w=csv.writer(buf);w.writerow(['arm','dataset','descriptor','condition','fusion','GT','N','baseline_R1','expanded_R1','gains','losses','delta']);w.writerows(rows);(W/'QUERY_RESULTS.csv').write_text(buf.getvalue())
    (W/'SUMMARY.md').write_text('# VPR run results\n\nResults below refer to the configured datasets, models and generated views.\n\nSee QUERY_RESULTS.csv, database/results/fresh__*.json, query/results/summary_fresh.json, analysis/results/statistical_families.json, analysis/gates/summary__fresh__*.json and SUBMISSIONS.json. All reported values are computed from this run.\n')

def execute(stage,index,shards):
    from .cli import COUNTS
    if stage=='prepare-data':prepare_data()
    elif stage=='audit-models':audit_models()
    elif stage in ['baseline','baseline-smoke']:
        import subprocess
        args=['--smoke'] if stage=='baseline-smoke' else ['--model',['CosPlace','EigenPlaces','AlexNet-GeM','ResNet50-GeM'][index//4],'--dataset',list(COUNTS)[index%4]]
        if stage=='baseline' and not 0<=index<16:raise ValueError('baseline index0..15')
        from .baselines import extract
        old_argv=sys.argv;sys.argv=['extract',*args]
        try:extract.main()
        finally:sys.argv=old_argv
    elif stage=='genwarp-smoke':
        from .generation import engine as generate
        request={'version':1,'run_id':'portable_smoke','sources':[]}
        for ds,role,schedule in [('GardensPoint','reference','gp_st_database_elevation_v1'),('Pittsburgh30k','query','pitts_realistic_v1')]:request['sources'].append({'dataset':ds,'role':role,'index':0,'id':C.real(ds,role)[0]['id'],'views':[{'schedule':schedule,'view_index':0,'repetitions':2}]})
        result=generate.run(request,execute=True,resume=True);C.write(W/'GENWARP_SMOKE.json',{'status':'pass','result':result})
    elif stage=='database-prepare':database_prepare()
    elif stage in ['database-pilot-generate','database-generate']:
        from .database import synthesis as generate_fresh
        generate_fresh.run(SimpleNamespace(pilot=stage=='database-pilot-generate',shard=index,shards=shards))
    elif stage in ['database-pilot-features','database-features']:
        from .database import features as descriptors
        descriptors.extract('fresh','gp' if stage=='database-pilot-features' else ['gp','st','pitts_elevation','pitts_realistic','tokyo'][index],stage=='database-pilot-features')
    elif stage in ['database-pilot-evaluate','database-evaluate']:
        from .database import retrieval as evaluate
        evaluate.run('fresh','gp' if stage=='database-pilot-evaluate' else ['gp','st','pitts_elevation','pitts_realistic','tokyo'][index],stage=='database-pilot-evaluate')
    elif stage=='query-prepare':query_prepare()
    elif stage in ['query-pilot-generate','query-generate']:
        from .query import synthesis as query_synthesis
        query_synthesis.run(SimpleNamespace(pilot=stage=='query-pilot-generate',shard=index,shards=shards))
    elif stage in ['query-null','query-pilot-features','query-features','query-merge']:
        from .query import features as query_features
        if stage=='query-null':query_features.nulls(index,shards)
        elif stage=='query-pilot-features':query_features.pilot()
        elif stage=='query-features':query_features.extract('fresh',index,shards)
        else:query_features.merge('fresh',shards)
    elif stage=='query-evaluate':
        from .query import retrieval as query_retrieval
        query_retrieval.run('fresh')
    elif stage=='robust-prepare':robust_prepare()
    elif stage in ['robust-features','robust-merge']:
        from .analysis import features as descriptor_features
        if stage=='robust-features':
            if not 0<=index<18:raise ValueError('robust-features index0..17')
            descriptor_features.extract(index)
        else:
            if not 0<=index<3:raise ValueError('robust-merge index0..2')
            descriptor_features.merge(index)
    elif stage=='robust-evaluate':
        from .analysis import retrieval as grid_retrieval
        if not 0<=index<8:raise ValueError('robust-evaluate index0..7')
        grid_retrieval.run(index)
    elif stage in ['gate-signals','gating']:
        from .analysis import gating as confidence_gating
        if stage=='gate-signals':confidence_gating.signal_stage()
        else:
            if not 0<=index<8:raise ValueError('gating index0..7')
            confidence_gating.run(index)
    elif stage=='statistics':
        from .analysis import comparisons as comparisons
        comparisons.c.ARMS=['fresh'];comparisons.stats();summary()
    elif stage=='plots':
        from .analysis.plots import plot_all
        plot_all()
    else:raise ValueError('unknown workflow stage')

def run_stage(stage,index=0,shards=6):
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Computation requires a real Slurm allocation')
    if index<0 or shards<1:raise ValueError('invalid index/shards')
    start=time.time();marker=W/'stage_status'/f'{stage}_{index}.json'
    before={'stage':stage,'index':index,'shards':shards,'job_id':os.getenv('SLURM_JOB_ID'),'status':'running'};C.write(marker,before)
    try:execute(stage,index,shards)
    except BaseException:
        C.write(marker,dict(before,status='failed',traceback=traceback.format_exc(),seconds=time.time()-start));raise
    else:C.write(marker,dict(before,status='completed',seconds=time.time()-start,exit_code=0))
