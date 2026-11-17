#!/usr/bin/env python3
"""Export all retained project code/configuration with no experiment payloads."""
import ast,datetime,hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nvs_vpr.packaging import sources
def content(path):return Path(path).read_bytes()
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    files=sources(ROOT);manifest={rel:hashlib.sha256(content(p)).hexdigest() for rel,p in files.items()};dst=ROOT/'distributions';dst.mkdir(exist_ok=True)
    log=dst/'PORTABLE_TESTS.txt'
    with log.open('w') as f:r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Portable tooling tests failed: '+str(log))
    shells=[]
    # All Python is parsed; original shell scripts are also syntax-checked without executing them.
    for rel,p in files.items():
        if p.suffix=='.py':ast.parse(p.read_text(),filename=rel)
        if p.suffix=='.sh':
            q=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
            if q.returncode:raise RuntimeError('Shell syntax failed '+rel+': '+q.stderr)
            shells.append(rel)
    extras={'distribution_validation/PORTABLE_TESTS.txt':log.read_bytes(),'SOURCE_FILE_INVENTORY.json':(json.dumps({'files':{rel:{'bytes':len(content(p)),'packaged_sha256':manifest[rel],'repository_original_sha256':sha(p),'application_source':True} for rel,p in files.items()},'excluded':['datasets/raw images','pretrained weights','generated images','descriptor/score matrices','runtime caches/results/logs','manuscripts','credentials and internal configuration','prior review ZIPs'],'scope':'functional VPR application, configuration, workflow scripts, tests and required source integrations'},indent=2)+'\n').encode()}
    for rel,b in extras.items():manifest[rel]=hashlib.sha256(b).hexdigest()
    top='nvs-vpr-eval';tmp=dst/'NVS_VPR_REPOSITORY.tmp.zip';archive=dst/'NVS_VPR_REPOSITORY.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,p in files.items():z.writestr(top+'/'+rel,content(p))
        for rel,b in extras.items():z.writestr(top+'/'+rel,b)
        z.writestr(top+'/SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    os.replace(tmp,archive)
    with tempfile.TemporaryDirectory(prefix='vpr_source_export_') as td:
        with zipfile.ZipFile(archive) as z:z.extractall(td)
        extracted=Path(td)/top
        result=subprocess.run([sys.executable,'-B','-I',str(extracted/'scripts/verify_repo.py'),str(extracted)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Fresh source ZIP verification failed '+result.stderr)
        # Execute the portable tests AGAIN from extracted code, with all input fixtures temporary.
        freshlog=dst/'FRESH_EXTRACTION_TESTS.txt'
        with freshlog.open('w') as f:t=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(extracted/'tests'),'-v'],cwd=extracted,stdout=f,stderr=subprocess.STDOUT)
        if t.returncode:raise RuntimeError('Extracted portable tooling tests failed '+str(freshlog))
    digest=sha(archive);(dst/'NVS_VPR_REPOSITORY.sha256').write_text(digest+'  '+archive.name+'\n')
    output={'status':'pass','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'source_files':len(files),'shell_scripts_syntax_checked':len(shells),'portable_tooling_tests':'passed in repository and fresh extraction','fresh_source_verification':json.loads(result.stdout),'GPU_experiments_rerun':False,'datasets_or_weights_included':False,'fresh_dependency_install_performed':False}
    (dst/'SOURCE_PACKAGE_VALIDATION.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
