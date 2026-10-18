"""
Example 2: Independent validators
=================================

Demonstrates running selected validators directly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validators import (
    AnalyticValidator,
    SpectralSumRuleValidator,
    OrthonormalityValidator,
    ErrorInjectionValidator,
)


def main():
    print('=' * 60)
    print('Example 2: Running independent validators')
    print('=' * 60)

    # Pick four representative validators
    validators_to_run = [
        ('Analytic checks',     AnalyticValidator()),
        ('Sum rules',           SpectralSumRuleValidator()),
        ('Orthonormality',      OrthonormalityValidator()),
        ('Error injection',     ErrorInjectionValidator()),
    ]

    for label, validator in validators_to_run:
        print(f'\n--- {label} ({type(validator).__name__}) ---')
        summary = validator.summary()
        print(f'   Tests: {summary["n_passed"]}/{summary["n_total"]} passed')
        for r in summary['individual_results']:
            mark = 'PASS' if r['passed'] else 'FAIL'
            err_keys = ('max_error', 'ground_error', 'max_residual',
                        'max_abs_diff', 'max_relative_error',
                        'commutator_norm', 'max_off_block_S')
            err = next((r[k] for k in err_keys if k in r), None)
            err_str = f' (err={err:.2e})' if err is not None else ''
            print(f'   [{mark}] {r["test_name"]}{err_str}')


if __name__ == '__main__':
    main()
