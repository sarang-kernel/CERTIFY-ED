#!/usr/bin/env python3
"""Reproduce everything: tests, experiments E0-E10, numbers and tables.

    python run.py                 quick mode (about 15 minutes on one core)
    python run.py --full          every experiment at paper size (several hours)
    python run.py --only E9,E10   selected experiments
    python run.py --help          all options (same as `certify-ed reproduce --help`)

Results go to results/run_<mode>_<timestamp>/ (manifest.json, SUMMARY.md, logs, records, paper/).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certify_ed.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["reproduce", *sys.argv[1:]]))
