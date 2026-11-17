"""CPU exhaustive max fusion over ORIGINAL real references, with portable sufficient evidence."""
import argparse,os,time
import numpy as np
from . import common as c
from . import features as f5
def retrieve(db,views):
 # GEMV per view and Baseline 4096-row chunks retain the exact original-view arithmetic.
 n=len(db);s=np.empty((9,n),dtype=np.float32)
 for v,q in enumerate(views):
  for start in range(0,n,4096):s[v,start:start+4096]=np.asarray(db[start:start+4096])@q
 per=[{'view_index':v-1,'indices':c.top(row)[0],'scores':c.top(row)[1]} for v,row in enumerate(s)]
 ix,sc=c.full_fusion(s)
 if c.portable_fusion(per)!=(ix,sc):raise ValueError('complete fusion differs from sufficient portable top10 union')
 return ix,sc,per
def run(arm):
 from ..generation import cache as cache_loader
 session=cache_loader.CacheSession();summary={};failure=set(c.failure_indices())
 for e in c.GROUPS:
  ds=e['dataset'];key=e['key'];rows=c.validate_rows(c.jsonl(c.catalog(arm,key)),e);path=c.QUERY/'cache'/f'{arm}__{key}.npy';m=c.read(path.with_suffix('.json'));syn=f5.load_features(path,f5.expected(rows,m['provenance']['model']));db=c.load_real(ds,'reference',session);q=c.load_real(ds,'query',session);queries=c.real(ds,'query');refs=c.real(ds,'reference');gt=c.read(c.ROOT/'baselines'/f'{ds}_gt.json');baseline=c.jsonl(c.QUERY/'evidence'/f'baseline__{ds}.jsonl.gz');source_indices=[s['index'] for s in c.sources(e)];pred=[];t=time.time()
  for j,i in enumerate(source_indices):
   rr=rows[j*8:j*8+8]
   if any(r['source_index']!=i or r['view_index']!=v for v,r in enumerate(rr)):raise ValueError('descriptor source/view block mismatch')
   ix,sc,per=retrieve(db,np.vstack([q[i],syn[j*8:j*8+8]]))
   if per[0]['indices']!=baseline[i]['reference_indices'] or per[0]['scores']!=baseline[i]['scores']:raise ValueError('original-view B0 mismatch')
   correctness={p:{str(k):bool(set(ix[:k])&set(g[i])) for k in [1,5,10]} for p,g in gt.items()}
   oracle={p:any(v['indices'][0] in g[i] for v in per[1:]) for p,g in gt.items()}
   record={'arm':arm,'configuration':key,'condition':e['condition'],'dataset':ds,'query_index':i,'query_id':queries[i]['id'],'reference_indices':ix,'reference_ids':[refs[k]['id'] for k in ix],'scores':sc,'view_top10':[dict(v,image_id=(queries[i]['id'] if v['view_index']==-1 else rr[v['view_index']]['image_id'])) for v in per],'correct':correctness,'oracle_synthetic_rank1':oracle,'selected_failure':ds=='Pittsburgh30k' and i in failure}
   pred.append(record)
   if j%200==0:print('RETRIEVE',arm,key,j,len(source_indices),flush=True)
  result={}
  for pop,selected in [('full',pred),('selected_failures',[r for r in pred if r['selected_failure']])]:
   if not selected:continue
   if pop=='full' and e['condition']=='zero':continue
   for radius in gt:
    b=[baseline[r['query_index']]['correct'][radius]['1'] for r in selected];a=[r['correct'][radius]['1'] for r in selected];oracle=[r['oracle_synthetic_rank1'][radius] for r in selected]
    result[pop+'__'+radius]={'n':len(selected),'baseline':{str(k):sum(baseline[r['query_index']]['correct'][radius][str(k)] for r in selected) for k in [1,5,10]},'max_fusion':{str(k):sum(r['correct'][radius][str(k)] for r in selected) for k in [1,5,10]},'oracle_synthetic_rank1':sum(oracle),'paired_baseline_to_max':c.paired(b,a),'paired_baseline_to_oracle':c.paired(b,oracle)}
  c.write_jsonl(c.QUERY/'results'/f'{arm}__{key}__predictions.jsonl.gz',pred)
  summary[key]={'dataset':ds,'condition':e['condition'],'arm':arm,'evaluations':result,'queries':len(pred),'seconds':time.time()-t,'feature_sha256':m['embedding_sha256']}
 c.write(c.QUERY/'results'/f'summary_{arm}.json',summary)
