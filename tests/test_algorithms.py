"""Numerical regression checks independent of GPU inference and private assets."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nvs_vpr.analysis import statistics as m,precision
from nvs_vpr.baselines import data
from nvs_vpr.generation import protocol
from nvs_vpr.query.ranking import union_top
class AlgorithmTests(unittest.TestCase):
 def test_max_fusion_and_tie_order(self):
  s=np.zeros((9,12),np.float32);s[0,3]=1;s[8,2]=1
  fused,weights=m.fuse(s);self.assertEqual(m.rank(fused['max'])[0][:2].tolist(),[2,3]);self.assertTrue(np.isfinite(weights).all())
 def test_mean_and_confidence_uniform_fallback(self):
  s=np.repeat(np.arange(9,dtype=np.float32)[:,None],12,axis=1);f,w=m.fuse(s);self.assertTrue(np.array_equal(f['conf'],f['mean']));self.assertTrue(np.all(w==0))
 def test_union_top10_matches_complete_max(self):
  s=np.random.default_rng(42).normal(size=(9,100)).astype(np.float32);tops=[]
  for row in s:
   ix,sc=m.rank(row);tops.append({'indices':ix.tolist(),'scores':sc.tolist()})
  ix,sc=union_top(tops);a,b=m.rank(s.max(axis=0));self.assertEqual(ix,a.tolist());self.assertEqual(sc,b.tolist())
 def test_recall_excludes_empty_gt(self):
  metric=data.metrics(np.array([[0,1],[1,0]]),[[0],[]]);self.assertEqual(metric['valid_queries'],1);self.assertEqual(metric['excluded_queries'],1);self.assertEqual(metric['recall']['1'],1)
 def test_seed_uses_original_global_source_identity(self):self.assertEqual(protocol.seed(20,7),42002007)
 def test_oracle_only_expands_rescued_failures(self):
  result,mask=m.oracle([1,0,1,0],[0,1,1,0]);self.assertEqual(mask.tolist(),[False,True,False,False]);self.assertEqual(result['fraction_requiring_expansion'],.25)
 def test_exact_paired_test_and_holm(self):
  pair=m.paired([0]*6,[1]*6);self.assertEqual(pair['mcnemar_raw_p'],.03125);self.assertTrue(np.allclose(m.holm([.01,.04,.03]),[.03,.06,.06]))
 def test_negative_entropy_and_no_synthesis_inputs(self):
  signal=m.signals(np.zeros((2,10),np.float32));self.assertTrue(np.allclose(signal['entropy'],-np.log(5)));self.assertTrue(np.all(signal['margin']==0))
 def test_float64_threshold_boundary_on_float32_signal(self):
  x=np.array([-1.609377384185791],np.float32);t=-1.6093773589668705;self.assertEqual(precision.below(x,t).tolist(),[True]);self.assertEqual(m.gate([0],[1],x,t).tolist(),[True])
 def test_holdout_cannot_choose_training_threshold(self):
  b=np.array([1,0,1,0,1,0],bool);e=~b;x=np.arange(6,dtype=np.float32);train=np.array([0,2,4]);before=precision.select_threshold(b,e,x,train);e[[1,3,5]]=~e[[1,3,5]];self.assertEqual(before,precision.select_threshold(b,e,x,train))
if __name__=='__main__':unittest.main()
