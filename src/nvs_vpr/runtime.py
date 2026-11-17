"""Separate read-only application sources from configured run outputs."""
import os
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[1]
ROOT=Path(os.environ.get('NVS_VPR_WORK_DIR',REPO/'work')).expanduser().resolve()
