"""Source inventory for the application repository."""
import os
from pathlib import Path

ROOT_FILES=['README.md','LICENSE','pyproject.toml','setup.py','requirements.txt','run.py','.gitignore']
DOCS={'THIRD_PARTY_NOTICES.md','MODULE_PROVENANCE.json'}
SUFFIXES={'.py','.json','.md','.txt','.sh','.toml','.yml','.yaml','.cpp','.cu','.h','.hpp','.cuh'}

def sources(root):
    root=Path(root);result={}
    for name in ROOT_FILES:
        p=root/name
        if p.is_file():result[name]=p
    for folder in ['src','configs','requirements','scripts','tests','integrations']:
        for current,dirs,files in os.walk(root/folder,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in ['__pycache__','.git','build','.pytest_cache'] and not (Path(current)/d).is_symlink())
            for name in files:
                p=Path(current)/name
                if p.is_symlink() or name=='generate_db_augmentation.py':continue
                if p.suffix in SUFFIXES or name in ['LICENSE','NOTICE']:
                    result[str(p.relative_to(root))]=p
    for name in DOCS:
        p=root/'docs'/name
        if p.is_file():result[str(p.relative_to(root))]=p
    return dict(sorted(result.items()))
