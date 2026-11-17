"""Exact paired tests and Holm correction for the configured retrieval and gate families."""
from . import common as c
from . import statistics as m
def stats():
 families=[]
 for arm in c.ARMS:
  for radius in ['radius25m','radius50m']:
   for contrast in ['GW_baseline','null_baseline','GW_vs_null']:
    entries=[]
    for ds in c.DATASETS:
     for model in c.MODELS:
      t=dict(arm=arm,dataset=ds,model=model);summary=c.read(c.ANALYSIS/'results'/f'summary__{c.key(t)}.json')
      for fusion in c.FUSIONS:
       condition='null' if contrast=='null_baseline' else 'viewpoint';pair=summary[condition+'__'+fusion+'__'+radius]['paired']
       if contrast=='GW_vs_null':
        a=c.jsonl(c.ANALYSIS/'results'/f'{c.key(t,"viewpoint")}__predictions.jsonl.gz');b=c.jsonl(c.ANALYSIS/'results'/f'{c.key(t,"null")}__predictions.jsonl.gz');pair=m.paired([r['correct'][radius][fusion]['1'] for r in b],[r['correct'][radius][fusion]['1'] for r in a])
       entries.append(dict(t,condition=condition,fusion=fusion,GT=radius,analysis='grid_'+contrast,raw_p=pair['mcnemar_raw_p'],gains=pair['gains'],losses=pair['losses'],delta=pair['delta']))
    adjusted=m.holm([r['raw_p'] for r in entries])
    for r,p in zip(entries,adjusted):r.update(holm_adjusted_p=p,reject_05=p<=.05)
    families.append({'name':f'{arm}__{radius}__grid_{contrast}','size':24,'method':'Holm FWER .05, exact two-sided McNemar','entries':entries})
   for condition in c.CONDITIONS:
    for analysis in ['fixed_signal','train_selected','twofold']:
     entries=[]
     for ds in c.DATASETS:
      for model in c.MODELS:
       t=dict(arm=arm,dataset=ds,model=model);gs=c.read(c.ANALYSIS/'gates'/f'summary__{c.key(t)}.json')
       for fusion in c.FUSIONS:
        g=gs[condition+'__'+fusion+'__'+radius]
        if analysis=='fixed_signal':items=[(r['signal'],r['fixed_signal_split_half']['paired']) for r in g['signals']]
        else:items=[(g['training_selected_signal']['signal'] if analysis=='train_selected' else 'out-of-fold',g['training_selected_signal']['paired'] if analysis=='train_selected' else g['twofold_sensitivity']['paired'])]
        for signal,pair in items:entries.append(dict(t,condition=condition,fusion=fusion,GT=radius,analysis='gate_'+analysis,signal=signal,raw_p=pair['mcnemar_raw_p'],gains=pair['gains'],losses=pair['losses'],delta=pair['delta']))
     adjusted=m.holm([r['raw_p'] for r in entries])
     for r,p in zip(entries,adjusted):r.update(holm_adjusted_p=p,reject_05=p<=.05)
     families.append({'name':f'{arm}__{radius}__{condition}__gate_{analysis}','size':72 if analysis=='fixed_signal' else 24,'method':'Holm FWER .05, two-sided exact McNemar; positive evidence also requires positive effect; crossfit sensitivity qualified separately','entries':entries})
 c.write(c.ANALYSIS/'results/statistical_families.json',families)
