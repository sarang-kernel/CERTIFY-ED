"""Conformance check of producer bundles without installing the console script: `python conformance/check_bundle.py DIR...`
(the same as `certify-ed conform`; see conformance/README.md)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certify_ed.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["conform", *sys.argv[1:]]))
