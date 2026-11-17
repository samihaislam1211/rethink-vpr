import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from nvs_vpr import packaging as archive,cli as project,pipeline as fresh
class PortableTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix='vpr_portable_test_');cls.base=Path(cls.temp.name);cls.cfg=json.loads((REPO/'configs/project.example.json').read_text());cls.cfg['work_dir']=str(cls.base/'run')
  for k in ['vpr_python','genwarp_python']:cls.cfg[k]=sys.executable
  for k in ['descriptor_checkpoints','genwarp_checkpoints','zoe_source','midas_source','depth_checkpoint']:cls.cfg[k]=str(cls.base/k)
  for ds,d in cls.cfg['datasets'].items():
   for role in d:d[role]=str(cls.base/'datasets'/ds/('GT.npz' if role=='gt_file' else role))
  with contextlib.redirect_stdout(io.StringIO()):project.prepare(cls.cfg)
  cls.work=Path(cls.cfg['work_dir']);fresh.initialize(cls.work)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test01_application_sources_are_separate_from_run_outputs(self):
  self.assertEqual(fresh.P,self.work);self.assertNotEqual(fresh.C.SOURCE,self.work);self.assertTrue((fresh.C.SOURCE/'baselines/extract.py').is_file());self.assertFalse((self.work/'src').exists())
 def test02_existing_run_cannot_be_overwritten(self):
  with self.assertRaises(FileExistsError):project.prepare(self.cfg)
 def test03_no_large_or_private_payloads_in_inventory(self):
  files=archive.sources(REPO);self.assertTrue(files);self.assertTrue(all(not set(Path(r).parts)&{'.git','.aws','.codex','cache','images','results','snapshots'} for r in files));self.assertTrue(all(Path(r).suffix not in ['.pth','.pt','.npy','.npz','.png','.jpg','.zip'] for r in files))
  self.assertIn('src/nvs_vpr/analysis/precision.py',files);self.assertIn('src/nvs_vpr/pipeline.py',files);self.assertFalse(any('paper_reimplementation' in f or 'reproduce/' in f for f in files));self.assertIn('integrations/genwarp_local/extract_latents.py',files)
 def test04_original_six_gpu_limit(self):
  bad=json.loads(json.dumps(self.cfg));bad['slurm']['max_gpus']=7
  with self.assertRaises(ValueError):project.validate(bad)
 def test05_missing_assets_reported_before_computation(self):self.assertTrue(project.validate(self.cfg,True))
 def test06_dataset_counts_and_original_gt(self):
  counts={'GardensPoint':(200,200),'StLucia':(200,200),'Pittsburgh30k':(1,1),'Tokyo247':(1,1)}
  for ds,(nr,nq) in counts.items():
   for role,n in [('reference',nr),('query',nq)]:
    folder=Path(self.cfg['datasets'][ds][role]);folder.mkdir(parents=True,exist_ok=True)
    for i in range(n):
     name=f'Image{i:03d}.jpg' if ds in ['GardensPoint','StLucia'] else '@0000100.00@0000200.00@17@T@0@0@id@0@.jpg'
     (folder/name).write_bytes(b'fixture; manifest test never decodes image')
  hard=np.zeros((200,200),bool);hard[np.arange(165),np.arange(165)]=True;soft=np.eye(200,dtype=bool);np.savez(self.cfg['datasets']['StLucia']['gt_file'],GThard=hard,GTsoft=soft)
  with patch.object(project,'COUNTS',counts):fresh.prepare_data()
  gt=fresh.C.read(fresh.P/'baselines/StLucia_gt.json');self.assertEqual(sum(bool(x) for x in gt['hard']),165);self.assertEqual(sum(bool(x) for x in gt['soft']),200)
  with patch.object(project,'COUNTS',dict(counts,Pittsburgh30k=(2,1))):
   with self.assertRaises(ValueError):fresh.prepare_data()
 def test07_source_ids_not_reassigned_for_database_selection(self):
  data=json.loads((REPO/'configs/selection/database_sources.json').read_text());self.assertEqual({k:len(v['sources']) for k,v in data.items()},{'gp':200,'st':200,'pitts_elevation':320,'pitts_realistic':320,'tokyo':2392});self.assertTrue(all('/reference/' in r['id'] for v in data.values() for r in v['sources']))
 def test08_array_scheduler_command_and_limit(self):
  from types import SimpleNamespace
  calls=[]
  def run(cmd,**kwargs):
   calls.append(cmd);output=''
   if cmd[0]=='sbatch':output='99999\n'
   elif cmd[:2]==['squeue','-h']:output='123|existing|gpu:1|R\n'
   return SimpleNamespace(returncode=0,stdout=output,stderr='')
  args=SimpleNamespace(stage='baseline',index=0,shards=6,array='0-15',after=None)
  with patch.object(project.subprocess,'run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):project.submit(self.cfg,args)
  cmd=next(c for c in calls if c[0]=='sbatch');self.assertIn('--array=0-15%5',cmd);self.assertIn('--gres=gpu:1',cmd);self.assertIn('${SLURM_ARRAY_TASK_ID}',Path(cmd[-1]).read_text());self.assertEqual(json.loads((self.work/'SUBMISSIONS.json').read_text())[-1]['stdout'],'99999')
 def test09_cpu_jobs_request_no_gpu(self):
  from types import SimpleNamespace
  calls=[]
  def run(cmd,**kwargs):calls.append(cmd);return SimpleNamespace(returncode=0,stdout='99998\n' if cmd[0]=='sbatch' else '',stderr='')
  args=SimpleNamespace(stage='prepare-data',index=0,shards=6,array=None,after=None)
  with patch.object(project.subprocess,'run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):project.submit(self.cfg,args)
  self.assertFalse(any(x.startswith('--gres') for c in calls if c[0]=='sbatch' for x in c))
 def test10_robustness_dispatches_fresh_tasks(self):
  from types import SimpleNamespace
  calls=[];mod=SimpleNamespace(extract=lambda i:calls.append(i))
  with patch.dict(sys.modules,{'nvs_vpr.analysis.features':mod}),patch('nvs_vpr.analysis.features',mod,create=True):
   for i in [0,6,12,17]:fresh.execute('robust-features',i,6)
  self.assertEqual(calls,[0,6,12,17])
 def test11_no_forged_slurm_allocation(self):
  import os
  with patch.dict(os.environ,{},clear=True):
   with self.assertRaises(RuntimeError):fresh.run_stage('prepare-data')
 def test12_all_workflow_stages_exist(self):
  import re
  stages=re.findall(r'^submit_wait ([a-z-]+)',(REPO/'scripts/run_pipeline.sh').read_text(),re.M);self.assertTrue(set(stages)<=set(project.STAGES))
 def test13_strict_fresh_cache_rejects_changed_checkpoint_embedding_and_source(self):
  import importlib.util,os
  from nvs_vpr.generation import cache as bridge
  core=fresh.C.base;dataset=next(d for d in fresh.C.read(fresh.P/'baselines/config.json')['datasets'] if d['name']=='GardensPoint');checkpoint=self.base/'fixture.pth';checkpoint.write_bytes(b'checkpoint fixture')
  model={'name':'Fixture','checkpoint':str(checkpoint),'sha256':core.sha(checkpoint),'dim':2};env={'test':'no inference'}
  expected=core.cache_meta(dataset,'reference',model,env);path=fresh.P/'baselines/cache/fixture.npy';array=np.zeros((200,2),np.float32);array[:,0]=1;np.save(path,array)
  fresh.C.write(path.with_suffix('.json'),{'fingerprint':core.fingerprint(expected),'provenance':expected,'embedding_sha256':core.sha(path),'shape':list(array.shape)})
  fresh.C.write(fresh.P/'baselines/results/GardensPoint__Fixture.json',{'model_spec':model})
  with patch.object(fresh.C,'current_env',return_value=env):
   self.assertTrue(np.array_equal(bridge.load(path,expected),array))
   checkpoint.write_bytes(b'changed checkpoint')
   with self.assertRaises(ValueError):bridge.load(path,expected)
   checkpoint.write_bytes(b'checkpoint fixture')
   original=path.read_bytes();array[:,0]=0;array[:,1]=1;np.save(path,array)
   with self.assertRaises(ValueError):bridge.load(path,expected)
   path.write_bytes(original)
   source=Path(core.manifest(dataset,'reference')[0]['path']);source.write_bytes(b'changed original source')
   with self.assertRaises(ValueError):bridge.load(path,expected)
 def test14_all_application_modules_import_without_external_audit_code(self):
  import importlib,pkgutil,nvs_vpr
  names=[m.name for m in pkgutil.walk_packages(nvs_vpr.__path__,nvs_vpr.__name__+'.')]
  for name in names:importlib.import_module(name)
  self.assertTrue(names);self.assertTrue(all('phase' not in name for name in names))
if __name__=='__main__':unittest.main(verbosity=2)
