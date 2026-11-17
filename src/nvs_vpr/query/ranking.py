"""Exact max-fusion top-K reconstruction from nine per-view top-K lists."""
import math
import numpy as np
def require(ok,why):
 if not ok:raise ValueError(why)
def union_top(tops):
 require(len(tops)==9,'nine views including real original required');candidates={}
 for v in tops:
  pairs=list(zip(v['indices'],v['scores']));require(len(pairs)==10 and len(set(v['indices']))==10 and np.isfinite(v['scores']).all(),'invalid per-view ranking')
  require(pairs==sorted(pairs,key=lambda z:(-z[1],z[0])),'view tie/rank order')
  for i,s in pairs:candidates[i]=max(candidates.get(i,-math.inf),s)
 ordered=sorted(candidates,key=lambda i:(-candidates[i],i))[:10];return ordered,[candidates[i] for i in ordered]
