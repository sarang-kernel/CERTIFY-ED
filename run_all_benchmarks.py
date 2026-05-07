#!/usr/bin/env python3
"""
CERTIFY-ED Master Validation Runner
====================================

Runs the complete CERTIFY-ED validation pipeline:
    1. pytest unit/integration test suite (53 tests)
    2. 13 independent validators across 16 physics models
    3. Aggregates all results and exports JSON + figures + tar.gz archive

Usage:
    python run_all_benchmarks.py [output_dir]

Outputs (under results/run_<timestamp>/):
    - manifest.json          : Index of all results
    - platform_info.json     : System info
    - pytest_output.txt      : Full pytest stdout
    - pytest_results.json    : Pytest summary
    - validators/*.json      : Per-validator detailed results
    - figures/*.png          : Visualizations
    - manuscript_data.json   : Aggregated key numbers for the paper
    - results.tar.gz         : Single-archive download
"""

import os
import sys
import json
import time
import tarfile
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

# Make sure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(msg: str, char: str = '=', width: int = 78) -> None:
    print(char * width)
    print(msg.center(width))
    print(char * width)
    sys.stdout.flush()


def section(msg: str) -> None:
    print()
    print('-' * 78)
    print(f'  {msg}')
    print('-' * 78)
    sys.stdout.flush()


def collect_platform_info() -> Dict[str, Any]:
    info = {
        'timestamp': datetime.now().isoformat(),
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
    }
    try:
        import scipy
        info['scipy_version'] = scipy.__version__
    except ImportError:
        pass
    try:
        import mpmath
        info['mpmath_version'] = mpmath.__version__
    except ImportError:
        info['mpmath_version'] = 'not installed'
    try:
        import quspin
        info['quspin_version'] = quspin.__version__
    except ImportError:
        info['quspin_version'] = 'not installed'
    return info


def run_pytest_suite(output_dir: str) -> Dict[str, Any]:
    """Run pytest, stream output to terminal, capture results to file."""
    section('STAGE 1: pytest test suite')
    print('Running pytest tests/ ... (output streamed to terminal)')
    print()
    sys.stdout.flush()

    pytest_log_path = os.path.join(output_dir, 'pytest_output.txt')
    pytest_result_path = os.path.join(output_dir, 'pytest_results.json')

    cmd = [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short',
           '--color=yes']
    start = time.time()

    # Run pytest, capturing AND printing
    output_lines: List[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            cwd=os.path.dirname(os.path.abspath(__file__)),
                            bufsize=1, universal_newlines=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
    proc.wait()
    elapsed = time.time() - start

    # Save log
    with open(pytest_log_path, 'w') as f:
        f.writelines(output_lines)

    # Parse summary - strip ANSI escape codes first
    import re
    full = ''.join(output_lines)
    full_clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', full)
    n_pass = full_clean.count(' PASSED')
    n_fail = full_clean.count(' FAILED')
    n_skip = full_clean.count(' SKIPPED')
    n_error = full_clean.count(' ERROR')
    summary = {
        'returncode': proc.returncode,
        'elapsed_seconds': elapsed,
        'passed': n_pass,
        'failed': n_fail,
        'skipped': n_skip,
        'errors': n_error,
        'success': proc.returncode == 0,
        'log_file': pytest_log_path,
    }
    with open(pytest_result_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f'pytest finished in {elapsed:.1f}s: {n_pass} passed, {n_fail} failed, '
          f'{n_skip} skipped, {n_error} errors')
    sys.stdout.flush()
    return summary


def run_all_validators(output_dir: str) -> Dict[str, Any]:
    """Run all 13 validators sequentially with terminal streaming."""
    section('STAGE 2: 13 independent validators')

    from validators import (
        AnalyticValidator, QuSpinValidator, HighPrecisionValidator,
        SparseDenseValidator, JordanWignerValidator, SpectralSumRuleValidator,
        OrthonormalityValidator, UnitarityValidator, ConservationLawValidator,
        SymmetrySectorValidator, ThermalLimitValidator,
        FiniteSizeScalingValidator, ErrorInjectionValidator,
    )

    validators_dir = os.path.join(output_dir, 'validators')
    os.makedirs(validators_dir, exist_ok=True)

    spec_list = [
        ('analytic', AnalyticValidator,
         'Closed-form analytic results (Bethe ansatz, SU(2), exact GS)'),
        ('quspin', QuSpinValidator,
         'Cross-validation against QuSpin'),
        ('high_precision', HighPrecisionValidator,
         'mpmath arbitrary-precision (50 digits) reference'),
        ('sparse_dense', SparseDenseValidator,
         'ARPACK Lanczos vs LAPACK direct'),
        ('jordan_wigner', JordanWignerValidator,
         'Free-fermion analytic spectra via JW'),
        ('sum_rules', SpectralSumRuleValidator,
         'Trace, Frobenius, log-det, operator norm sum rules'),
        ('orthonormality', OrthonormalityValidator,
         'Eigenvector orthonormality + completeness + spectral decomp'),
        ('unitarity', UnitarityValidator,
         'exp(-iHt) unitarity and group property'),
        ('conservation', ConservationLawValidator,
         'Conservation laws [H,S]=0 and quantum-number block-diag'),
        ('symmetry_sectors', SymmetrySectorValidator,
         'Symmetry-resolved spectrum decomposition'),
        ('thermal', ThermalLimitValidator,
         'High-T (beta->0) and low-T (beta->inf) thermal limits'),
        ('scaling', FiniteSizeScalingValidator,
         'Finite-size scaling toward thermodynamic limits'),
        ('error_injection', ErrorInjectionValidator,
         'Framework error-detection capability'),
    ]

    aggregate = {}
    overall_total = 0
    overall_pass = 0

    for vid, vcls, desc in spec_list:
        print()
        print(f'>> [{vid}] {desc}')
        sys.stdout.flush()
        start = time.time()
        try:
            v = vcls()
            summary = v.summary()
            n_total = summary.get('n_total', 0)
            n_pass = summary.get('n_passed', 0)
            available = summary.get('available', True)
            elapsed = time.time() - start

            if not available:
                status = 'SKIPPED'
            elif n_total == 0:
                status = 'EMPTY'
            elif n_pass == n_total:
                status = 'PASS'
            else:
                status = 'PARTIAL'

            print(f'   {status}: {n_pass}/{n_total} tests, {elapsed:.1f}s')
            if status == 'PARTIAL':
                for r in summary.get('individual_results', []):
                    if not r.get('passed', True):
                        print(f'     FAIL  {r.get("test_name", "?")}')
            sys.stdout.flush()

            # Save per-validator JSON
            out_path = os.path.join(validators_dir, f'{vid}_results.json')
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            aggregate[vid] = {
                'description': desc,
                'n_total': n_total,
                'n_passed': n_pass,
                'available': available,
                'elapsed_seconds': elapsed,
                'status': status,
                'output_file': out_path,
            }
            if available:
                overall_total += n_total
                overall_pass += n_pass

        except Exception as e:
            print(f'   ERROR: {type(e).__name__}: {e}')
            sys.stdout.flush()
            aggregate[vid] = {
                'description': desc,
                'status': 'ERROR',
                'error': f'{type(e).__name__}: {e}',
            }

    print()
    print(f'Validator suite total: {overall_pass}/{overall_total} tests passed')
    sys.stdout.flush()

    return {
        'overall_total': overall_total,
        'overall_passed': overall_pass,
        'per_validator': aggregate,
    }


def generate_figures(output_dir: str, validator_results: Dict[str, Any]) -> List[str]:
    """Generate manuscript figures from validator results."""
    section('STAGE 3: Generating figures')

    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    figure_paths: List[str] = []

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping figures')
        return []

    # Load validator JSONs we need
    val_dir = os.path.join(output_dir, 'validators')

    def load_json(name):
        path = os.path.join(val_dir, f'{name}_results.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # Figure 1: residual heatmap across models from sum rules / orthonormality
    sum_rules = load_json('sum_rules')
    if sum_rules:
        fig, ax = plt.subplots(figsize=(9, 5))
        results = sum_rules.get('individual_results', [])
        models = [r['test_name'].replace('sum_rules_', '') for r in results]
        errs = [
            [r.get('trace_error', 0), r.get('trace_squared_error', 0),
             r.get('trace_cubed_error', 0), r.get('operator_norm_error', 0)]
            for r in results
        ]
        errs = np.array(errs).T
        # log10 with floor
        errs_log = np.log10(np.maximum(errs, 1e-18))
        im = ax.imshow(errs_log, aspect='auto', cmap='viridis_r', vmin=-16, vmax=-10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticks(range(4))
        ax.set_yticklabels(['tr(H)', 'tr(H²)', 'tr(H³)', '||H||₂'])
        plt.colorbar(im, label='log₁₀(error)')
        ax.set_title('Sum Rule Errors Across Models')
        plt.tight_layout()
        path = os.path.join(figures_dir, 'fig1_sum_rule_errors.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        figure_paths.append(path)
        print(f'   wrote {path}')

    # Figure 2: finite-size scaling (Heisenberg, free fermion, TFIM critical)
    scaling = load_json('scaling')
    if scaling:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        results = scaling.get('individual_results', [])
        for ax, r in zip(axes, results):
            sizes = r.get('sizes', [])
            if 'energies_per_bond' in r:
                vals = r['energies_per_bond']
                limit = r.get('bethe_ansatz_limit')
                ylab = r'$E_0 / N$ (per bond)'
            elif 'energies_per_site' in r:
                vals = r['energies_per_site']
                limit = r.get('thermodynamic_limit')
                ylab = r'$E_0 / N$ (per site)'
            else:
                continue
            ax.plot(sizes, vals, 'o-', label='Computed')
            if limit is not None:
                ax.axhline(limit, color='red', ls='--',
                          label=f'Limit = {limit:.4f}')
            ax.set_xlabel('N')
            ax.set_ylabel(ylab)
            ax.set_title(r['test_name'].replace('_', ' ').title())
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(figures_dir, 'fig2_finite_size_scaling.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        figure_paths.append(path)
        print(f'   wrote {path}')

    # Figure 3: validator pass/fail summary
    fig, ax = plt.subplots(figsize=(10, 6))
    val_aggregate = validator_results.get('per_validator', {})
    names = list(val_aggregate.keys())
    counts_pass = [val_aggregate[n].get('n_passed', 0) for n in names]
    counts_total = [val_aggregate[n].get('n_total', 0) for n in names]
    counts_fail = [t - p for t, p in zip(counts_total, counts_pass)]
    y = np.arange(len(names))
    ax.barh(y, counts_pass, color='tab:green', label='passed')
    ax.barh(y, counts_fail, left=counts_pass, color='tab:red', label='failed')
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Number of tests')
    ax.set_title('Validator Test Counts')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(figures_dir, 'fig3_validator_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    figure_paths.append(path)
    print(f'   wrote {path}')

    # Figure 4: error injection - all should be detected
    err_inj = load_json('error_injection')
    if err_inj:
        fig, ax = plt.subplots(figsize=(9, 5))
        results = err_inj.get('individual_results', [])
        names = [r['test_name'].replace('_', '\n', 1) for r in results]
        detected = [1 if r.get('detected', r.get('passed', False)) else 0
                    for r in results]
        colors = ['tab:green' if d else 'tab:red' for d in detected]
        ax.bar(range(len(names)), detected, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylim([0, 1.2])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Missed', 'Detected'])
        ax.set_title('Error Injection: All Errors Must Be Detected')
        plt.tight_layout()
        path = os.path.join(figures_dir, 'fig4_error_injection.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        figure_paths.append(path)
        print(f'   wrote {path}')

    return figure_paths


def aggregate_manuscript_data(output_dir: str, pytest_summary: Dict,
                                validator_results: Dict,
                                platform_info: Dict) -> str:
    """Produce manuscript_data.json with aggregated key numbers."""
    section('STAGE 4: Aggregating manuscript data')

    val_dir = os.path.join(output_dir, 'validators')
    val_data = {}
    for fname in sorted(os.listdir(val_dir)):
        if fname.endswith('_results.json'):
            with open(os.path.join(val_dir, fname)) as f:
                val_data[fname.replace('_results.json', '')] = json.load(f)

    summary = {
        'package': 'CERTIFY-ED',
        'version': '1.0.0',
        'timestamp': platform_info['timestamp'],
        'platform': platform_info,
        'pytest': pytest_summary,
        'validators': {
            'total_individual_tests': validator_results['overall_total'],
            'total_individual_tests_passed': validator_results['overall_passed'],
            'per_validator_summary': validator_results['per_validator'],
        },
        'headline_numbers': {
            'n_models_supported': 16,
            'n_independent_validators': 13,
            'n_pytest_tests': pytest_summary.get('passed', 0)
                              + pytest_summary.get('failed', 0),
            'pytest_passed': pytest_summary.get('passed', 0),
            'validator_tests_passed': validator_results['overall_passed'],
            'validator_tests_total': validator_results['overall_total'],
        },
    }

    out_path = os.path.join(output_dir, 'manuscript_data.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'   wrote {out_path}')
    return out_path


def make_archive(output_dir: str) -> str:
    """Create tar.gz archive of all outputs."""
    section('STAGE 5: Creating archive')
    archive_path = output_dir + '.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    print(f'   wrote {archive_path}')
    return archive_path


def main():
    output_root = sys.argv[1] if len(sys.argv) > 1 else 'results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_root, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    banner('CERTIFY-ED MASTER VALIDATION RUN')
    print(f'Output directory: {output_dir}')
    print()

    platform_info = collect_platform_info()
    with open(os.path.join(output_dir, 'platform_info.json'), 'w') as f:
        json.dump(platform_info, f, indent=2)
    print('Platform:')
    for k, v in platform_info.items():
        print(f'  {k}: {v}')

    overall_start = time.time()

    pytest_summary = run_pytest_suite(output_dir)
    validator_results = run_all_validators(output_dir)
    figures = generate_figures(output_dir, validator_results)
    manuscript_path = aggregate_manuscript_data(
        output_dir, pytest_summary, validator_results, platform_info
    )

    # Write manifest
    manifest = {
        'output_dir': output_dir,
        'timestamp': platform_info['timestamp'],
        'pytest_log': os.path.join(output_dir, 'pytest_output.txt'),
        'pytest_results': os.path.join(output_dir, 'pytest_results.json'),
        'validators_dir': os.path.join(output_dir, 'validators'),
        'figures': figures,
        'manuscript_data': manuscript_path,
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    archive = make_archive(output_dir)

    overall = time.time() - overall_start
    print()
    banner('RUN COMPLETE')
    print(f'  Total time:                {overall:.1f}s')
    print(f'  pytest:                    {pytest_summary.get("passed", 0)}/'
          f'{pytest_summary.get("passed", 0) + pytest_summary.get("failed", 0)} passed')
    print(f'  Validator individual tests: {validator_results["overall_passed"]}/'
          f'{validator_results["overall_total"]} passed')
    print(f'  Output directory:          {output_dir}')
    print(f'  Archive:                   {archive}')

    # Exit code: 0 if all green, 1 otherwise
    success = (pytest_summary.get('success', False)
               and validator_results['overall_passed']
                   == validator_results['overall_total'])
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
