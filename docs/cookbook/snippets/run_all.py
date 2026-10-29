"""Re-run every cookbook snippet in a scratch directory and print its output (to refresh the docs pages)."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["quickstart.py", "read_certificate.py", "numpy_eigh.py", "eigsh_lowest.py", "files_import.py",
         "quspin.py", "matrix_free.py", "producer_bundle.py", "shift_invert_window.py", "inconclusive_examples.py"]

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        for name in ORDER:
            r = subprocess.run([sys.executable, os.path.join(HERE, name)], capture_output=True, text=True, cwd=d)
            print(f"===== {name} (exit {r.returncode})")
            print(r.stdout, end="")
            if r.returncode:
                print(r.stderr, file=sys.stderr)
