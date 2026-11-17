"""Frozen nine-representation fusion, first-pass gating and paired inference."""
import functools,math
import numpy as np
FUSIONS=['max','mean','conf'];SIGNALS=['margin','top1','entropy']
def rank(s,k=10):
 x=np.asarray(s)
 if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('invalid reference similarity vector')
 ix=np.lexsort((np.arange(len(x)),-x))[:k];return ix,x[ix]
def fuse(s):
 a=np.asarray(s,dtype=np.float32)
 if a.ndim!=2 or a.shape[0]!=9 or a.shape[1]<2 or not np.isfinite(a).all():raise ValueError('need finite original plus eight view scores')
 parts=np.partition(a,-2,axis=1)[:,-2:];w=(parts[:,1]-parts[:,0]).astype(np.float32);total=w.sum(dtype=np.float32)
 out={'max':a.max(axis=0),'mean':a.mean(axis=0,dtype=np.float32)}
 out['conf']=out['mean'].copy() if total==0 else (w[:,None]*a).sum(axis=0,dtype=np.float32)/total
 return out,w
def signals(top10):
 a=np.asarray(top10,dtype=np.float32)
 if a.ndim!=2 or a.shape[1]<5 or not np.isfinite(a).all() or np.any(np.diff(a,axis=1)>0):raise ValueError('need descending raw-query scores')
 p=np.exp(a[:,:5]-a[:,0,None]);p/=p.sum(axis=1,keepdims=True)
 return {'margin':a[:,0]-a[:,1],'top1':a[:,0].copy(),'entropy':(p*np.log(p+1e-12)).sum(axis=1)}
def thresholds(x,n=200):
 x=np.asarray(x)
 if not len(x) or not np.isfinite(x).all():raise ValueError('invalid confidence population')
 qs=np.unique(np.quantile(x,np.linspace(0,1,n)));eps=(x.max()-x.min())*1e-6+1e-12
 return np.r_[-np.inf,qs+eps,np.inf]
def encode_tau(t):return '-inf' if t==-np.inf else ('+inf' if t==np.inf else float(t))
def decode_tau(t):return -np.inf if t=='-inf' else (np.inf if t=='+inf' else float(t))
def split(n,seed=42):
 if n<2:raise ValueError('split requires at least two queries')
 p=np.random.default_rng(seed).permutation(n);return p[:n//2],p[n//2:]
def select_threshold(b,e,x,train):
 from .precision import select_threshold as select
 return select(b,e,x,train)
def gate(b,e,x,t):
 from .precision import below
 return np.where(below(x,t),np.asarray(e,bool),np.asarray(b,bool))
def mcnemar(b,a):
 b=np.asarray(b,bool);a=np.asarray(a,bool);gain=int((~b&a).sum());loss=int((b&~a).sum());d=gain+loss
 p=min(1.,2*sum(math.comb(d,i) for i in range(min(gain,loss)+1))/2**d) if d else 1.
 return gain,loss,p
@functools.lru_cache(maxsize=100000)
def interval(n,g,l,reps=10000,seed=42):
 draws=np.random.default_rng(seed).multinomial(n,[l/n,(n-l-g)/n,g/n],size=reps);return tuple(float(x) for x in np.quantile((draws[:,2]-draws[:,0])/n,[.025,.975]))
def paired(b,a,reps=10000):
 b=np.asarray(b,bool);a=np.asarray(a,bool)
 if b.shape!=a.shape or b.ndim!=1 or not len(b):raise ValueError('paired population mismatch')
 n=len(b);g,l,p=mcnemar(b,a)
 return {'n':n,'baseline_successes':int(b.sum()),'expanded_successes':int(a.sum()),'gains':g,'losses':l,'retained_correct':int((b&a).sum()),'retained_incorrect':int((~b&~a).sum()),'delta':(g-l)/n,'mcnemar_raw_p':p,'CI95':list(interval(n,g,l,reps)),'bootstrap_seed':42,'bootstrap_replicates':reps,'bootstrap_method':'paired query differences; exact-equivalent multinomial distribution; percentile95'}
def oracle(b,e):
 b=np.asarray(b,bool);e=np.asarray(e,bool);use=~b&e
 return {'baseline_successes':int(b.sum()),'recoverable_failures':int(use.sum()),'oracle_successes':int((b|e).sum()),'n':len(b),'oracle_R1':float((b|e).mean()),'absolute_improvement':float(use.mean()),'fraction_requiring_expansion':float(use.mean())},use
def holm(pvalues):
 ps=np.asarray(pvalues,float)
 if ps.ndim!=1 or not np.isfinite(ps).all() or np.any((ps<0)|(ps>1)):raise ValueError('invalid p values')
 order=np.argsort(ps,kind='stable');adj=np.minimum(1,np.maximum.accumulate((len(ps)-np.arange(len(ps)))*ps[order]));out=np.empty(len(ps));out[order]=adj;return out.tolist()
