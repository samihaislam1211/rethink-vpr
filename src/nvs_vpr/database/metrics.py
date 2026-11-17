"""Primary source-label metrics; independent verifier does not import this module."""
import math
import numpy as np
K=(1,5,10)
def wilson(success,n):
 if n==0:return None
 z=1.959963984540054;p=success/n;d=1+z*z/n;centre=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0.,centre-half),min(1.,centre+half)]
def aggregate(source_predictions,synthetic,gt,indices,baseline):
 valid=[j for j in indices if gt[j]];excluded=[j for j in indices if not gt[j]]
 result={'population_queries':len(indices),'valid_queries':len(valid),'excluded_queries':len(excluded),'excluded_query_indices':excluded,'numerators':{},'recall':{},'gains':{},'losses':{},'delta_recall':{}}
 for k in K:
  current={j:bool(set(source_predictions[j,:k])&set(gt[j])) for j in valid};old={j:bool(set(baseline[j,:k])&set(gt[j])) for j in valid}
  result['numerators'][str(k)]=sum(current.values());result['recall'][str(k)]=sum(current.values())/len(valid) if valid else None
  result['gains'][str(k)]=sum(current[j] and not old[j] for j in valid);result['losses'][str(k)]=sum(old[j] and not current[j] for j in valid)
  result['delta_recall'][str(k)]=(result['gains'][str(k)]-result['losses'][str(k)])/len(valid) if valid else None
 result['successful_queries']=result['numerators']['1'];result['failed_queries']=len(valid)-result['successful_queries'];result['synthetic_top1_count']=sum(bool(synthetic[j,0]) for j in valid)
 result['incorrect_synthetic_top1_count']=sum(bool(synthetic[j,0]) and int(source_predictions[j,0]) not in gt[j] for j in valid)
 result['synthetic_occupancy_rate']=result['synthetic_top1_count']/len(valid) if valid else None;result['pollution_rate']=result['incorrect_synthetic_top1_count']/len(valid) if valid else None
 result['R1_Wilson95']=wilson(result['successful_queries'],len(valid));result['pollution_Wilson95']=wilson(result['incorrect_synthetic_top1_count'],len(valid));return result

def gp_components(source_predictions,synthetic):
 if len(source_predictions)!=200:raise ValueError('GP population must contain200 original queries')
 total=adjacent=genuine=0;rows=[]
 for j in range(200):
  source=int(source_predictions[j,0]);gap=abs(source-j);synth=bool(synthetic[j,0]);p=synth and gap!=0;a=synth and 1<=gap<=8;g=synth and gap>8
  total+=p;adjacent+=a;genuine+=g;rows.append({'query_index':j,'source_frame_index':source,'synthetic_top1':synth,'exact_correct':gap==0,'absolute_frame_gap':gap,'within8':gap<=8,'exact_pollution':p,'adjacent_artifact':a,'genuine_outside8':g})
 if total!=adjacent+genuine:raise AssertionError('GP partition failed')
 return {'denominator':200,'exact_pollution_count':total,'adjacent_count':adjacent,'genuine_count':genuine,'exact_pollution_rate':total/200,'adjacent_rate':adjacent/200,'genuine_rate':genuine/200},rows
