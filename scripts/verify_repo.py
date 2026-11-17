#!/usr/bin/env python3
"""Dependency-free integrity and source syntax verification after ZIP extraction."""
import argparse,ast,hashlib,json
from pathlib import Path

def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'SOURCE_SHA256.json').read_text());checks=0
    for rel,wanted in manifest.items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or not p.is_file():raise ValueError('missing/unsafe source: '+rel)
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=wanted:raise ValueError('changed source: '+rel)
        checks+=1
        if p.suffix=='.py':ast.parse(p.read_text(),filename=rel);checks+=1
        if p.suffix=='.ipynb':
            notebook=json.loads(p.read_text())
            if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook['cells'] if c.get('cell_type')=='code'):raise ValueError('saved notebook outputs included')
            checks+=1
    return {'status':'pass','checks':checks,'failures':0,'source_files':len(manifest),'scope':'complete source-artifact SHA256 and Python syntax; no dataset/weights/model-inference claim'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',nargs='?',default='.');a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
