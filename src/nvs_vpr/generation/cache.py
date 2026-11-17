"""Fresh-run strict cache loader. Historical relocation exceptions are unavailable."""
from pathlib import Path
import numpy as np
from . import protocol as c
from ..baselines import data as core
NEW=core.sha(core.SOURCE/'baselines/extract.py')
CORE=core.sha(core.SOURCE/'baselines/data.py')
class CacheSession:
    def __init__(self):self.checked={}
def load(path,expected,*,session=None,decision_dir=None):
    path=Path(path);meta=c.read(path.with_suffix('.json'));original=meta['provenance']
    ds=next(d for d in c.read(core.ROOT/'baselines/config.json')['datasets'] if d['name']==original['dataset'])
    from ..database.common import current_env
    current=core.cache_meta(ds,original['role'],original['model'],current_env())
    if current!=expected or original!=current:raise ValueError('fresh cache source/model/input/environment changed; recompute')
    result=c.read(core.ROOT/'baselines/results'/f"{original['dataset']}__{original['model']['name']}.json")
    if result['model_spec']!=original['model']:raise ValueError('fresh cache descriptor differs from baseline result')
    model=current['model'];session=session or CacheSession();checkpoint=Path(model['checkpoint']);key=(str(checkpoint),checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    if key not in session.checked:session.checked[key]=core.sha(checkpoint)
    if session.checked[key]!=model['sha256']:raise ValueError('checkpoint bytes changed')
    a=core.load_cache(path,current)
    if a.dtype!=np.float32 or list(a.shape)!=meta['shape'] or a.shape[1]!=model['dim']:raise ValueError('fresh cache dimension/dtype changed')
    if decision_dir:_decision(decision_dir,path,meta)
    return a
def _decision(directory,path,meta):
    Path(directory).mkdir(parents=True,exist_ok=True)
    c.write(Path(directory)/(path.stem+'.json'),{'accepted':True,'scope':'strict fresh cache; no historical compatibility exception','embedding_sha256':meta['embedding_sha256'],'fingerprint':meta['fingerprint']})
