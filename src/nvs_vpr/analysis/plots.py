"""Generate result figures from this run's JSON/curve data."""
from pathlib import Path
import json
import numpy as np
from ..runtime import ROOT

def plot_all():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=ROOT/'analysis/figures';output.mkdir(parents=True,exist_ok=True);provenance={}
    def read(p):return json.loads(p.read_text())
    # All full-population query effects, including matched conventional controls.
    paths=sorted((ROOT/'analysis/results').glob('summary__fresh__*.json'))
    if not paths:raise FileNotFoundError('Run descriptor/fusion evaluation before plotting')
    rows=[r for path in paths for r in read(path).values() if r['GT']=='radius25m']
    figure,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,ds in zip(axes,['Pittsburgh30k','Tokyo247']):
        labels=sorted({r['model'] for r in rows if r['dataset']==ds})
        for condition,style in [('viewpoint','o'),('null','x')]:
            for j,fusion in enumerate(['max','mean','conf']):
                data=[next(r for r in rows if r['dataset']==ds and r['model']==m and r['condition']==condition and r['fusion']==fusion) for m in labels]
                ax.scatter([100*r['paired']['delta'] for r in data],np.arange(len(labels))+.12*(j-1),marker=style,label=condition+'/'+fusion)
        ax.axvline(0,color='gray',linewidth=.7);ax.set_yticks(range(len(labels)),labels);ax.set_title(ds);ax.set_xlabel('Change in Recall@1 (percentage points)');ax.legend(fontsize=7)
    figure.tight_layout();figure.savefig(output/'query_fusion.png',dpi=180);plt.close(figure);provenance['query_fusion.png']=[str(p.relative_to(ROOT)) for p in paths]
    # Original-only confidence gate sweeps, with full-sweep interpretation in the labels.
    from ..query.common import jsonl
    figure,axes=plt.subplots(1,2,figsize=(10,4));curve_paths=[]
    for ax,ds in zip(axes,['Pittsburgh30k','Tokyo247']):
        path=ROOT/'analysis/gates/curves'/f'fresh__{ds}__CosPlace__viewpoint__max__radius25m.jsonl.gz'
        curve=jsonl(path);curve_paths.append(str(path.relative_to(ROOT)))
        for sig in ['margin','top1','entropy']:
            data=[r for r in curve if r['signal']==sig];ax.plot([r['coverage'] for r in data],[r['recall1'] for r in data],label=sig)
        ax.set_title(ds+' CosPlace/max');ax.set_xlabel('Fraction expanded');ax.set_ylabel('Recall@1');ax.legend()
    figure.suptitle('Exploratory full sweeps; held-out decisions are reported separately');figure.tight_layout();figure.savefig(output/'confidence_gates.png',dpi=180);plt.close(figure);provenance['confidence_gates.png']=curve_paths
    # Database metrics use the full query population and the declared primary GT.
    figure,axes=plt.subplots(1,2,figsize=(11,4));dbpaths=[]
    for key,gt in [('gp','exact'),('st','hard'),('pitts_elevation','radius25m'),('pitts_realistic','radius25m'),('tokyo','radius25m')]:
        budgets=[0,1,2,4,8];paths=[ROOT/'database/results'/f'fresh__{key}__B{b}.json' for b in budgets];values=[read(p)['metrics']['full'][gt] for p in paths];dbpaths.extend(str(p.relative_to(ROOT)) for p in paths)
        axes[0].plot(budgets,[v['recall']['1'] for v in values],'o-',label=key);axes[1].plot(budgets,[v['pollution_rate'] for v in values],'o-',label=key)
    for ax,title in zip(axes,['Recall@1','Incorrect synthetic top-1 fraction']):ax.set_xlabel('Views per selected reference');ax.set_ylabel(title);ax.legend(fontsize=7)
    figure.tight_layout();figure.savefig(output/'database_budgets.png',dpi=180);plt.close(figure);provenance['database_budgets.png']=dbpaths
    (output/'SOURCES.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance
