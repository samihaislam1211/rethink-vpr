"""Explicit comparison precision; keep frozen float32 signals and threshold grid.

Python float and numpy.float64 scalars can have different promotion behavior in
NumPy 2 when compared with float32 arrays. Promote BOTH operands to float64 so
the exact saved threshold has the same strict-inequality meaning everywhere.
"""
import numpy as np
from . import statistics as statistics
def below(confidence,threshold):
 x=np.asarray(confidence)
 if not np.isfinite(x).all() or np.isnan(np.float64(threshold)):raise ValueError('nonfinite confidence or NaN threshold')
 return np.less(x.astype(np.float64),np.float64(threshold))
def select_threshold(baseline,expanded,confidence,training):
 b=np.asarray(baseline,bool);a=np.asarray(expanded,bool);x=np.asarray(confidence);tr=np.asarray(training,int)
 if not len(tr) or len(np.unique(tr))!=len(tr) or tr.min()<0 or tr.max()>=len(x):raise ValueError('invalid training indices')
 thresholds=statistics.thresholds(x[tr]);success=[int(np.where(below(x[tr],t),a[tr],b[tr]).sum()) for t in thresholds];j=int(np.argmax(success))
 return float(thresholds[j]),success[j],len(thresholds)
