#!/usr/bin/env python3
"""
Validation Experiment 3: Cross-Platform Reproducibility (Paper §6.3)

Tests bitwise-identical certificate generation across platforms.
Expected: Identical SHA-256 hashes.

Author: Sarang Vehale
"""

from certifyEd.validation import run_cross_platform_test
import sys


def main():
    results = run_cross_platform_test()

    print("\n" + "=" * 80)
    if results["match"]:
        print("✓ Validation Experiment 3 PASSED")
    else:
        print("✗ Validation Experiment 3 FAILED")
    print("=" * 80)

    return 0 if results["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
