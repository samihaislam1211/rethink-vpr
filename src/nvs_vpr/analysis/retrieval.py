"""All-reference nine-view scoring and full max/mean/conf grid, checkpointed by query block."""
import argparse,hashlib,os,time
from pathlib import Path
import numpy as np
from . import common as c
from . import statistics as m
BLOCK=64
def similarities(db,views):
 a=np.empty((9,len(db)),np.float32)
 if views.shape[0]!=9:raise ValueError('original plus eight representations required')
 for v,q in enumerate(views):
  for start in range(0,len(db),4096):a[v,start:start+4096]=np.asarray(db[start:start+4096])@q
 return a
def certificate(s,fused,weights):
 raw=[m.rank(v) for v in s];ranked=[m.rank(fused[name]) for name in c.FUSIONS];candidates=sorted(set(int(i) for ix,sc in ranked for i in ix));count=len(candidates);ci=np.full(30,-1,np.int32);ci[:count]=candidates;cv=np.zeros((9,30),np.float32);cv[:,:count]=s[:,candidates];outside=np.ones(s.shape[1],bool);outside[candidates]=False;outside_indices=np.flatnonzero(outside);excluded_scores=[];excluded_indices=[]
 for name,(ix,sc) in zip(c.FUSIONS,ranked):
  values=fused[name][outside];j=int(np.argmax(values));excluded_scores.append(values[j]);excluded_indices.append(outside_indices[j])
  if (-float(values[j]),int(outside_indices[j]))<=(-float(sc[-1]),int(ix[-1])):raise ValueError('excluded reference outranks claimed top10')
 return {'candidate_indices':ci,'candidate_count':np.int16(count),'candidate_scores':cv,'weights':weights,'raw_top_indices':np.array([x[0] for x in raw],np.int32),'raw_top_scores':np.array([x[1] for x in raw],np.float32),'ranked_indices':np.array([x[0] for x in ranked],np.int32),'ranked_scores':np.array([x[1] for x in ranked],np.float32),'excluded_max':np.array(excluded_scores,np.float32),'excluded_index':np.array(excluded_indices,np.int32)}
def paths(t,condition,start):
 stem=f'{c.key(t,condition)}__block{start:05d}';return c.ANALYSIS/'cache/scores'/f'{stem}.npy',c.ANALYSIS/'evidence/fusion'/f'{stem}.npz',c.ANALYSIS/'results/blocks'/f'{stem}.jsonl.gz',c.ANALYSIS/'evidence/score_blocks'/f'{stem}.json'
def run(index):
 from ..generation import cache as cache_loader
 c.require_slurm();t=c.task(index);ds=t['dataset'];base=c.baseline(ds,t['model']);queries=c.real(ds,'query');gt=c.gt(ds);session=cache_loader.CacheSession();db=c.load_real(ds,'reference',t['model'],session);q=c.load_real(ds,'query',t['model'],session);summary={};job_start=time.time()
 for condition in c.CONDITIONS:
  rows,synthetic,featurepath=c.load_synthetic(t,condition);signature=c.fp({'task':t,'condition':condition,'embedding_sha256':c.sha(featurepath),'ordered_images_sha256':c.ordered_hash(rows),'real_reference_sha256':c.read(c.ROOT/'baselines/cache'/f'{ds}__{t["model"]}__reference.json')['embedding_sha256'],'real_query_sha256':c.read(c.ROOT/'baselines/cache'/f'{ds}__{t["model"]}__query.json')['embedding_sha256'],'model':c.read(featurepath.with_suffix('.json'))['provenance']['model'],'math_sha256':c.sha(c.SOURCE/'analysis/statistics.py'),'evaluation_sha256':c.sha(__file__),'GT_sha256':c.sha(c.ANALYSIS/'manifests'/f'GT__{ds}.npz')});allpred=[];times={'similarity_seconds':0.,'fusion_seconds':{x:0. for x in c.FUSIONS},'ranking_certificate_seconds':0.};tiny=0
  query=c.jsonl(c.ANALYSIS/'evidence/query_predictions'/f'{t["arm"]}__{ds}__{condition}__predictions.jsonl.gz') if t['model']=='CosPlace' else None
  for start in range(0,len(q),BLOCK):
   stop=min(start+BLOCK,len(q));sp,cp,pp,mp=paths(t,condition,start)
   if mp.exists():
    meta=c.read(mp)
    if meta['fingerprint']!=signature or any(c.sha(p)!=meta[k] for p,k in [(sp,'scores_sha256'),(cp,'certificate_sha256'),(pp,'predictions_sha256')]):raise ValueError('incompatible/corrupt completed score block')
    pred=c.jsonl(pp);allpred.extend(pred)
    for name in ['similarity_seconds','ranking_certificate_seconds']:times[name]+=meta['timing'][name]
    for name in c.FUSIONS:times['fusion_seconds'][name]+=meta['timing']['fusion_seconds'][name]
    tiny+=meta['nonzero_margin_sum_below_historical_epsilon'];continue
   if any(p.exists() for p in [sp,cp,pp]):raise ValueError('untracked output block; no successful-output overwrite')
   block=np.empty((stop-start,9,len(db)),np.float32);certs=[];pred=[];bt={'similarity_seconds':0.,'fusion_seconds':{x:0. for x in c.FUSIONS},'ranking_certificate_seconds':0.};btiny=0
   for offset,i in enumerate(range(start,stop)):
    rr=rows[i*8:i*8+8]
    if [(r['source_index'],r['view_index']) for r in rr]!=[(i,v) for v in range(8)]:raise ValueError('synthetic source/view order mismatch')
    tick=time.perf_counter();s=similarities(db,np.vstack([q[i],synthetic[i*8:i*8+8]]));bt['similarity_seconds']+=time.perf_counter()-tick;block[offset]=s
    # Time each rule directly, retaining historical float32 arithmetic.
    tick=time.perf_counter();fm=s.max(axis=0);bt['fusion_seconds']['max']+=time.perf_counter()-tick
    tick=time.perf_counter();mean=s.mean(axis=0,dtype=np.float32);bt['fusion_seconds']['mean']+=time.perf_counter()-tick
    tick=time.perf_counter();part=np.partition(s,-2,axis=1)[:,-2:];weights=(part[:,1]-part[:,0]).astype(np.float32);total=weights.sum(dtype=np.float32);conf=mean.copy() if total==0 else (weights[:,None]*s).sum(axis=0,dtype=np.float32)/total;bt['fusion_seconds']['conf']+=time.perf_counter()-tick;btiny+=int(0<total<=1e-12)
    tick=time.perf_counter();fused={'max':fm,'mean':mean,'conf':conf};cert=certificate(s,fused,weights);certs.append(cert);bt['ranking_certificate_seconds']+=time.perf_counter()-tick
    if cert['raw_top_indices'][0].tolist()!=base[i]['reference_indices'] or cert['raw_top_scores'][0].tolist()!=base[i]['scores']:raise ValueError('exact Baseline original-query B0 regression')
    if query is not None:
     old=query[i]
     if old['query_id']!=queries[i]['id'] or cert['ranked_indices'][0].tolist()!=old['reference_indices'] or cert['ranked_scores'][0].tolist()!=old['scores']:raise ValueError('Query expansion CosPlace max regression')
    rankings={name:{'reference_indices':cert['ranked_indices'][j].tolist(),'scores':cert['ranked_scores'][j].tolist()} for j,name in enumerate(c.FUSIONS)}
    correct={radius:{name:{str(k):bool(set(rankings[name]['reference_indices'][:k])&set(g[i])) for k in [1,5,10]} for name in c.FUSIONS} for radius,g in gt.items()}
    pred.append({'arm':t['arm'],'dataset':ds,'model':t['model'],'condition':condition,'query_index':i,'query_id':queries[i]['id'],'rankings':rankings,'correct':correct,'complete_score_row_sha256':hashlib.sha256(s.tobytes()).hexdigest()})
    if i==0:c.save_npz(c.ANALYSIS/'evidence/representative_scores'/f'{c.key(t,condition)}.npz',scores=s,query_index=np.array(i),query_id=np.array(queries[i]['id']))
   c.save_array(sp,block);c.save_npz(cp,**{name:np.array([r[name] for r in certs]) for name in certs[0]});c.write_jsonl(pp,pred,exclusive=True);c.write(mp,{'task':t,'condition':condition,'start':start,'stop':stop,'fingerprint':signature,'scores_path':str(sp),'scores_sha256':c.sha(sp),'certificate_path':str(cp.relative_to(c.ANALYSIS)),'certificate_sha256':c.sha(cp),'predictions_path':str(pp.relative_to(c.ANALYSIS)),'predictions_sha256':c.sha(pp),'timing':bt,'nonzero_margin_sum_below_historical_epsilon':btiny,'job_id':os.getenv('SLURM_JOB_ID')},exclusive=True);allpred.extend(pred);tiny+=btiny
   for name in ['similarity_seconds','ranking_certificate_seconds']:times[name]+=bt[name]
   for name in c.FUSIONS:times['fusion_seconds'][name]+=bt['fusion_seconds'][name]
   print('SCORE_BLOCK',c.key(t,condition),start,stop,flush=True)
  c.write_jsonl(c.ANALYSIS/'results'/f'{c.key(t,condition)}__predictions.jsonl.gz',allpred)
  for name in c.FUSIONS:
   for radius in gt:
    b=[r['correct'][radius]['1'] for r in base];a=[r['correct'][radius][name]['1'] for r in allpred];pair=m.paired(b,a);oracle,mask=m.oracle(b,a);nums={str(k):sum(r['correct'][radius][name][str(k)] for r in allpred) for k in [1,5,10]};bn={str(k):sum(r['correct'][radius][str(k)] for r in base) for k in [1,5,10]}
    summary[condition+'__'+name+'__'+radius]={'arm':t['arm'],'dataset':ds,'model':t['model'],'condition':condition,'fusion':name,'GT':radius,'n':len(q),'baseline_numerators':bn,'expanded_numerators':nums,'baseline_recall':{k:v/len(q) for k,v in bn.items()},'expanded_recall':{k:v/len(q) for k,v in nums.items()},'recall_difference':{k:(nums[k]-bn[k])/len(q) for k in nums},'paired':pair,'oracle_gate':oracle}
  c.write(c.ANALYSIS/'evidence/timing'/f'{c.key(t,condition)}.json',{'task':t,'condition':condition,'queries':len(q),'timing':times,'nonzero_tiny_margin_sum_cases':tiny,'reference_count':len(db),'new_generation_seconds':0})
 c.write(c.ANALYSIS/'results'/f'summary__{c.key(t)}.json',summary);c.write(c.ANALYSIS/'evidence/evaluations'/f'{c.key(t)}.json',{'status':'pass','job_id':os.getenv('SLURM_JOB_ID'),'seconds':time.time()-job_start,'conditions':2,'fusion_rules':3,'GTs':2})
