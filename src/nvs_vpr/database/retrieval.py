"""Database augmentation CPU image-level ranking and primary metrics, using unchanged Baseline topk."""
import argparse,os,time
from pathlib import Path
import numpy as np
from . import common as c
from . import metrics as m


def run(arm,key,pilot=False):
 if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Retrieval requires allocated Slurm CPUs')
 if not c.read(c.DATABASE/'evidence/B0_equivalence.json')['passed']:raise ValueError('B0 gate not passed')
 from . import features as d
 cfg=c.config(key);name=cfg['dataset'];rows=d.rows_for(arm,key,pilot);suffix='__pilot' if pilot else '';path=c.DATABASE/'cache'/f'{arm}__{key}{suffix}.npy';spec=c.read(c.ROOT/'baselines/results'/f'{name}__CosPlace.json')['model_spec'];features=d.load_features(path,d.cache_expected(rows,spec))
 real_r,query=c.real(name,'reference'),c.real(name,'query');r,q=c.load_real(name,'reference'),c.load_real(name,'query');base_rows=c.jsonl(c.DATABASE/'evidence'/f'B0_recomputed__{name}.jsonl.gz');baseline=np.array([x['reference_indices'] for x in base_rows]);base_scores=np.array([x['scores'] for x in base_rows]);gt=c.read(c.ROOT/'baselines'/f'{name}_gt.json')
 catalog=[{'index':i,'image_id':x['id'],'source_index':i,'source_id':x['id'],'synthetic':False,'view_index':None} for i,x in enumerate(real_r)]
 catalog.extend({'index':len(real_r)+i,'image_id':x['image_id'],'source_index':x['source_index'],'source_id':x['source_id'],'synthetic':True,'view_index':x['view_index']} for i,x in enumerate(rows))
 catalog_name=f'{arm}__{key}{suffix}';c.write_jsonl(c.DATABASE/'results'/f'{catalog_name}__catalog.jsonl.gz',catalog,exclusive=True)
 populations={'full':list(range(len(query)))}
 if name=='Pittsburgh30k':
  sample=c.read(c.DATABASE/'config'/f'cohort__{key}.json')['sampled_query_indices'];expected=sorted(np.random.default_rng(42).choice(6816,60,replace=False).tolist())
  if sample!=expected:raise ValueError('historical60-query cohort differs')
  populations['historical60']=sample
 for b in ([0,1,2] if pilot else [0,1,2,4,8]):
  started=time.perf_counter();mask=np.array([x['view_index']<b for x in rows]);selected=np.flatnonzero(mask);canonical=np.concatenate([np.arange(len(r)),len(r)+selected])
  if b==0:ix,scores=baseline.copy(),base_scores.copy();retrieval_seconds=next(x['retrieval_seconds'] for x in c.read(c.DATABASE/'evidence/B0_equivalence.json')['datasets'] if x['dataset']==name);reused=True
  else:
   db=np.concatenate([np.asarray(r),features[selected]],axis=0);local,scores=c.base.topk(db,q);ix=canonical[local];retrieval_seconds=time.perf_counter()-started;reused=False
  source=np.array([[catalog[i]['source_index'] for i in a] for a in ix]);synth=np.array([[catalog[i]['synthetic'] for i in a] for a in ix]);metrics={population:{g:m.aggregate(source,synth,lists,indices,baseline) for g,lists in gt.items()} for population,indices in populations.items()}
  stem=f'{catalog_name}__B{b}';pred=[]
  for j,queryrow in enumerate(query):
   pred.append({'query_index':j,'query_id':queryrow['id'],'image_indices':ix[j].tolist(),'source_indices':source[j].tolist(),'synthetic':synth[j].tolist(),'scores':scores[j].tolist(),'correct':{g:{str(k):bool(set(source[j,:k])&set(lists[j])) if lists[j] else None for k in [1,5,10]} for g,lists in gt.items()}})
  pred_path=c.DATABASE/'results'/f'{stem}__predictions.jsonl.gz';c.write_jsonl(pred_path,pred,exclusive=True)
  result={'arm':arm,'arm_label':c.read(c.DATABASE/'config/protocol.json')['arms'][arm],'configuration':key,'dataset':name,'descriptor':'CosPlace','budget':b,'budget_unit':'views per selected reference image','pilot':pilot,'augmented_source_count':len({row['source_index'] for row in rows}),'synthetic_indexed':len(selected),'real_reference_count':len(r),'index_entries':len(canonical),'synthetic_image_bytes':sum(rows[i]['integrity']['bytes'] for i in selected),'descriptor_index_payload_bytes':len(canonical)*2048*4,'rank_protocol':'image-level no spatial NMS; score descending then real-first canonical image index ascending','ranking_deduplicated':False,'query_count':len(query),'populations':populations,'metrics':metrics,'predictions_file':pred_path.name,'predictions_sha256':c.sha(pred_path),'catalog_file':catalog_name+'__catalog.jsonl.gz','catalog_sha256':c.sha(c.DATABASE/'results'/f'{catalog_name}__catalog.jsonl.gz'),'GT_file':f'GT__{name}.npz','GT_sha256':c.sha(c.DATABASE/'manifests'/f'GT__{name}.npz'),'synthetic_cache_fingerprint':c.read(path.with_suffix('.json'))['fingerprint'],'real_cache_decisions':{role:c.read(c.ROOT/'baselines/cache'/f'{name}__CosPlace__{role}.json')['embedding_sha256'] for role in ['reference','query']},'retrieval_seconds':retrieval_seconds,'B0_ranking_reused_from_verified_recomputation':reused,'job_id':os.getenv('SLURM_JOB_ID'),'evaluation_sha256':c.sha(__file__),'metrics_source_sha256':c.sha(c.SOURCE/'database/metrics.py')}
  if name=='GardensPoint':
   components,individual=m.gp_components(source,synth);result['GP_decomposition']=components
   for j,row in enumerate(individual):row.update(top1_image_id=catalog[ix[j,0]]['image_id'],top1_catalog_index=int(ix[j,0]),query_id=query[j]['id'])
   c.write_jsonl(c.DATABASE/'results'/f'{stem}__GP_decomposition.jsonl.gz',individual,exclusive=True)
  c.write(c.DATABASE/'results'/f'{stem}.json',result,exclusive=True);print('EVALUATED',arm,key,b,{p:{g:(x['numerators']['1'],x['valid_queries'],x['incorrect_synthetic_top1_count']) for g,x in ms.items()} for p,ms in metrics.items()},flush=True)
 if pilot:c.write(c.DATABASE/'evidence/pilot_end_to_end.json',{'passed':True,'generation':c.read(c.DATABASE/'evidence/pilot_generation.json'),'descriptor_cache':str(path),'evaluated_budgets':[0,1,2],'queries':len(query),'selected_sources':1,'scope':'partial GP diagnostic only, not full primary experiment','job_id':os.getenv('SLURM_JOB_ID')})

