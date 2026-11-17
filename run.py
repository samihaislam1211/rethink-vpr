#!/usr/bin/env python3
"""Run the VPR application from a checkout without installation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from nvs_vpr.cli import main
if __name__=='__main__':main()
