#!/usr/bin/env python3
"""
Validation Experiment 1: Bethe Ansatz Comparison (Paper §6.1)

Validates CERTIFY-ED against exact analytical solutions.
Expected: 15-16 digit agreement for small systems.

Author: Sarang Vehale
"""

from certifyEd.validation import run_bethe_ansatz_validation
import sys


def main():
    results = run_bethe_ansatz_validation()

    print("\n" + "=" * 80)
    print("✓ Validation Experiment 1 COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
