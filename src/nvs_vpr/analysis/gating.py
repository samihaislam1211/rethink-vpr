"""Original-only signals, frozen thresholds, honest holdout and train-only signal choice."""
import argparse,os,time
import numpy as np
from . import common as c
from . import statistics as m
from .precision import below,select_threshold
def signal_stage():
 c.require_slurm();out=[]
 for ds in c.DATASETS:
  for model in c.MODELS:
   base=c.baseline(ds,model);start=time.perf_counter();values=m.signals(np.asarray([r['scores'] for r in base],np.float32));seconds=time.perf_counter()-start
   c.save_npz(c.ANALYSIS/'gates/signals'/f'{ds}__{model}.npz',**values)
   meta={'dataset':ds,'model':model,'source':'original real-query Baseline top-five scores only','baseline_sha256':c.sha(c.ANALYSIS/'evidence'/f'baseline__{ds}__{model}.jsonl.gz'),'signal_sha256':c.sha(c.ANALYSIS/'gates/signals'/f'{ds}__{model}.npz'),'signal_calculation_seconds':seconds,'n':len(base),'no_GT_or_synthetic_inputs':True};out.append(meta)
 c.write(c.ANALYSIS/'evidence/signals.json',{'status':'pass','entries':out})
def summary_one(b,a,confidence):
 b=np.asarray(b,bool);a=np.asarray(a,bool);n=len(b);train,test=m.split(n);sigrows=[];curves=[];best_masks=[];half_masks=[];training_scores=[]
 for signal in c.SIGNALS:
  x=confidence[signal];ts=m.thresholds(x);rows=[]
  for tau in ts:
   mask=below(x,tau);flags=np.where(mask,a,b);g,l,p=m.mcnemar(b,flags)
   row={'signal':signal,'tau':m.encode_tau(tau),'n':n,'expanded':int(mask.sum()),'coverage':float(mask.mean()),'successes':int(flags.sum()),'recall1':float(flags.mean()),'delta':float(flags.mean()-b.mean()),'gains':g,'losses':l,'CI95':list(m.interval(n,g,l,2000))};rows.append(row)
  best=int(np.argmax([r['successes'] for r in rows]));fulltau=m.decode_tau(rows[best]['tau']);fullmask=below(x,fulltau);best_masks.append(fullmask)
  tau,train_correct,nt=select_threshold(b,a,x,train);halfmask=below(x,tau);half_masks.append(halfmask);training_scores.append(train_correct);flags=np.where(halfmask,a,b)
  sigrows.append({'signal':signal,'biased_best_of_full_sweep':{'tau':m.encode_tau(fulltau),'curve_index':best,'coverage':float(fullmask.mean()),'paired':m.paired(b,np.where(fullmask,a,b)),'selection':'optimistically biased; not confirmatory'},'fixed_signal_split_half':{'tau':m.encode_tau(tau),'training_successes':train_correct,'training_n':len(train),'candidate_thresholds':nt,'heldout_n':len(test),'heldout_expanded':int(halfmask[test].sum()),'heldout_coverage':float(halfmask[test].mean()),'paired':m.paired(b[test],flags[test]),'selection':'threshold selected using training half only; fixed signal prespecified'}});curves.extend(rows)
 chosen=int(np.argmax(training_scores));signal=c.SIGNALS[chosen];tau=m.decode_tau(sigrows[chosen]['fixed_signal_split_half']['tau']);selectedmask=below(confidence[signal],tau);selectedflags=np.where(selectedmask,a,b)
 selected={'signal':signal,'tau':m.encode_tau(tau),'training_successes':training_scores[chosen],'signal_selection':'training successes only; fixed margin,top1,entropy tie order','heldout_coverage':float(selectedmask[test].mean()),'heldout_expanded':int(selectedmask[test].sum()),'paired':m.paired(b[test],selectedflags[test])}
 # Reproduce the historical secondary-selection summary only as an explicit bias diagnostic.
 historical_j=int(np.argmax([r['fixed_signal_split_half']['paired']['delta'] for r in sigrows]));historical={'signal':c.SIGNALS[historical_j],'heldout_delta':sigrows[historical_j]['fixed_signal_split_half']['paired']['delta'],'label':'secondary-selection biased: selected best signal using held-out outcomes; not an honest gate'}
 # A second prespecified fold trains on B and evaluates A; every query is predicted out of fold.
 candidates=[select_threshold(b,a,confidence[s],test) for s in c.SIGNALS];j=int(np.argmax([z[1] for z in candidates]));other_signal=c.SIGNALS[j];other_tau=candidates[j][0];xfmask=np.zeros(n,bool);xfmask[test]=selectedmask[test];xfmask[train]=below(confidence[other_signal][train],other_tau);xf=np.where(xfmask,a,b)
 cross={'folds':[{'train':'A','evaluate':'fresh','signal':signal,'tau':m.encode_tau(tau)},{'train':'fresh','evaluate':'A','signal':other_signal,'tau':m.encode_tau(other_tau)}],'coverage':float(xfmask.mean()),'paired':m.paired(b,xf),'label':'additional twofold out-of-fold train-only signal/threshold selection sensitivity; primary fixed-half results remain separate'}
 og,oracle_mask=m.oracle(b,a)
 arrays={'baseline_correct':b,'expanded_correct':a,'train_indices':train,'test_indices':test,'biased_full_sweep_masks':np.array(best_masks,bool),'fixed_signal_masks':np.array(half_masks,bool),'train_selected_mask':selectedmask,'twofold_mask':xfmask,'oracle_gate_mask':oracle_mask}
 return {'n':n,'split_seed':42,'signals':sigrows,'training_selected_signal':selected,'twofold_sensitivity':cross,'historical_heldout_best_signal_biased':historical,'oracle_gate':og,'expand_none':m.paired(b,b),'expand_all':m.paired(b,a)},curves,arrays
def run(index):
 c.require_slurm();t=c.task(index);base=c.baseline(t['dataset'],t['model']);confidence={}
 with np.load(c.ANALYSIS/'gates/signals'/f'{t["dataset"]}__{t["model"]}.npz',allow_pickle=False) as z:confidence={s:z[s] for s in c.SIGNALS}
 out={}
 for condition in c.CONDITIONS:
  pred=c.jsonl(c.ANALYSIS/'results'/f'{c.key(t,condition)}__predictions.jsonl.gz')
  if [r['query_id'] for r in pred]!=[r['query_id'] for r in base]:raise ValueError('gating population mismatch')
  for fusion in c.FUSIONS:
   for radius in ['radius25m','radius50m']:
    b=[r['correct'][radius]['1'] for r in base];a=[r['correct'][radius][fusion]['1'] for r in pred];start=time.perf_counter();result,curve,arrays=summary_one(b,a,confidence);result['analysis_seconds']=time.perf_counter()-start
    start=time.perf_counter();np.where(below(confidence['margin'],0.),np.asarray(a,bool),np.asarray(b,bool));result['measured_vectorized_gate_seconds']=time.perf_counter()-start
    label=f'{c.key(t,condition)}__{fusion}__{radius}';c.write_jsonl(c.ANALYSIS/'gates/curves'/f'{label}.jsonl.gz',curve);c.save_npz(c.ANALYSIS/'gates/decisions'/f'{label}.npz',**arrays);out[condition+'__'+fusion+'__'+radius]=result
    print('GATING_COMPLETE',label,flush=True)
 c.write(c.ANALYSIS/'gates'/f'summary__{c.key(t)}.json',out)
