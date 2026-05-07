#!/usr/bin/env python
"""
CERTIFY-ED Master Benchmark Runner
==================================

This script runs ALL benchmarks, tests, and validations, then packages
the results for manuscript inclusion.

Usage:
    python run_all_benchmarks.py [--quick] [--no-tests] [--no-quspin]

Output:
    results/                        # Directory with all outputs
        manifest.json               # Summary of all runs
        platform_info.json          # Platform/system information
        analytic_results.json       # Analytic benchmark results
        quspin_results.json         # QuSpin cross-validation
        performance_results.json    # Performance benchmarks
        platform_results.json       # Platform reproducibility
        error_injection_results.json # Error detection tests
        test_output.txt             # pytest output
        manuscript_data.json        # Aggregated data for manuscript
        figures/                    # Generated plots (if matplotlib)
    
    certify_ed_results_TIMESTAMP.tar.gz  # Single archive with everything

After running:
    Send the .tar.gz archive for manuscript revision.
"""

import os
import sys
import json
import time
import tarfile
import subprocess
import argparse
import platform
import traceback
from datetime import datetime
from pathlib import Path

# Make sure we can import certify_ed
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))


def print_header(text: str, char: str = "=") -> None:
    print()
    print(char * 70)
    print(f"  {text}")
    print(char * 70)


def print_section(text: str) -> None:
    print()
    print("-" * 70)
    print(f"  {text}")
    print("-" * 70)


def safe_run(name: str, func, *args, **kwargs):
    """Run a function and capture errors gracefully."""
    print(f"\n>>> Running: {name}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"    [DONE] {name} completed in {elapsed:.2f}s")
        return {'status': 'success', 'result': result, 'elapsed_seconds': elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    [FAIL] {name} failed: {e}")
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed_seconds': elapsed
        }


def run_pytest(results_dir: Path) -> dict:
    """Run pytest test suite."""
    print_section("Running test suite (pytest)")
    
    test_output_file = results_dir / "test_output.txt"
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        with open(test_output_file, 'w') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        # Parse pytest output
        lines = result.stdout.splitlines()
        passed = failed = errors = 0
        for line in lines:
            if " passed" in line and " in " in line.lower():
                # Summary line like "5 passed, 1 failed in 0.5s"
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "passed" and i > 0:
                        try:
                            passed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif p == "failed" and i > 0:
                        try:
                            failed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif p == "error" or p == "errors":
                        try:
                            errors = int(parts[i-1])
                        except ValueError:
                            pass
        
        return {
            'returncode': result.returncode,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'all_passed': result.returncode == 0,
            'output_file': str(test_output_file.name)
        }
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'returncode': -1, 'all_passed': False}
    except Exception as e:
        return {'status': 'exception', 'error': str(e), 'all_passed': False}


def run_analytic_benchmarks() -> dict:
    """Run analytic validation benchmarks."""
    from benchmarks.analytic_benchmarks import AnalyticBenchmarks
    
    bench = AnalyticBenchmarks(tolerance=1e-12)
    bench.run_all()
    return bench.summary()


def run_quspin_validation() -> dict:
    """Run QuSpin cross-validation."""
    from benchmarks.quspin_validation import QuSpinValidator
    
    validator = QuSpinValidator(tolerance=1e-10)
    validator.run_all()
    return validator.summary()


def run_performance_benchmarks(quick: bool = False) -> dict:
    """Run performance benchmarks."""
    from benchmarks.performance_benchmarks import PerformanceBenchmarks
    
    if quick:
        bench = PerformanceBenchmarks(n_runs=3, max_n_sites=8)
    else:
        bench = PerformanceBenchmarks(n_runs=5, max_n_sites=10)
    
    bench.run_all()
    return bench.summary()


def run_platform_benchmarks() -> dict:
    """Run platform reproducibility benchmarks."""
    from benchmarks.platform_benchmarks import PlatformBenchmarks
    
    bench = PlatformBenchmarks()
    bench.run_all()
    return bench.summary()


def run_error_injection_tests() -> dict:
    """Run error injection tests."""
    from benchmarks.error_injection import ErrorInjectionTests
    
    tests = ErrorInjectionTests()
    tests.run_all()
    return tests.summary()


def generate_figures(results_dir: Path, all_results: dict) -> dict:
    """Generate matplotlib figures if available."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    figures_generated = []
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Figure 1: Performance scaling
        try:
            perf_data = all_results.get('performance', {}).get('result', {})
            if perf_data:
                scaling_result = perf_data.get('individual_results', [{}])[0]
                if scaling_result.get('test_name') == 'performance_scaling':
                    sizes_data = scaling_result['sizes']
                    
                    Ns = [s['n_sites'] for s in sizes_data]
                    single_means = [s['single_oracle_time']['mean_ms'] for s in sizes_data]
                    single_stds = [s['single_oracle_time']['std_ms'] for s in sizes_data]
                    multi_means = [s['multi_oracle_time']['mean_ms'] for s in sizes_data]
                    multi_stds = [s['multi_oracle_time']['std_ms'] for s in sizes_data]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.errorbar(Ns, single_means, yerr=single_stds, 
                                marker='o', label='Single Oracle (NumPy)', capsize=3)
                    ax.errorbar(Ns, multi_means, yerr=multi_stds,
                                marker='s', label='Multi-Oracle Consensus', capsize=3)
                    ax.set_xlabel('System size N (qubits)')
                    ax.set_ylabel('Time (ms)')
                    ax.set_yscale('log')
                    ax.set_title('CERTIFY-ED Performance Scaling')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    fig_path = figures_dir / "performance_scaling.png"
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figures_generated.append(str(fig_path.name))
                    print(f"    Generated: performance_scaling.png")
        except Exception as e:
            print(f"    Could not generate performance plot: {e}")
        
        # Figure 2: Residuals across analytic benchmarks
        try:
            analytic = all_results.get('analytic', {}).get('result', {})
            if analytic:
                results_list = analytic.get('individual_results', [])
                names = [r['test_name'] for r in results_list]
                residuals = [r['max_residual'] for r in results_list]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(names)), residuals, color='steelblue')
                ax.set_yscale('log')
                ax.axhline(y=1e-12, color='red', linestyle='--', label='Tolerance: 1e-12')
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([n.replace('_', '\n') for n in names], 
                                   rotation=0, ha='center', fontsize=8)
                ax.set_ylabel('Max Residual')
                ax.set_title('Residuals Across Analytic Benchmarks')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                fig_path = figures_dir / "analytic_residuals.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                figures_generated.append(str(fig_path.name))
                print(f"    Generated: analytic_residuals.png")
        except Exception as e:
            print(f"    Could not generate analytic residuals plot: {e}")
        
        # Figure 3: Oracle disagreement levels
        try:
            analytic = all_results.get('analytic', {}).get('result', {})
            if analytic:
                results_list = analytic.get('individual_results', [])
                names = [r['test_name'] for r in results_list]
                disagreements = [r['max_disagreement'] for r in results_list]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(names)), disagreements, color='darkgreen')
                ax.set_yscale('log')
                ax.axhline(y=1e-10, color='orange', linestyle='--', label='Consensus tolerance')
                ax.axhline(y=1e-15, color='red', linestyle=':', label='Machine epsilon')
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([n.replace('_', '\n') for n in names],
                                   rotation=0, ha='center', fontsize=8)
                ax.set_ylabel('Max Oracle Disagreement')
                ax.set_title('Multi-Oracle Consensus Quality')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                fig_path = figures_dir / "consensus_quality.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                figures_generated.append(str(fig_path.name))
                print(f"    Generated: consensus_quality.png")
        except Exception as e:
            print(f"    Could not generate consensus plot: {e}")
        
    except ImportError:
        print("    matplotlib not available, skipping figures")
    
    return {'generated_figures': figures_generated, 'figures_directory': str(figures_dir.name)}


def create_manuscript_data(all_results: dict) -> dict:
    """Aggregate results into manuscript-ready format."""
    
    # Extract key numbers for manuscript
    manuscript_data = {
        'generation_time': datetime.now().isoformat(),
        'summary': {},
        'tables': {},
        'key_numbers': {}
    }
    
    # Analytic validation summary
    if all_results.get('analytic', {}).get('status') == 'success':
        analytic = all_results['analytic']['result']
        manuscript_data['summary']['analytic'] = {
            'tests_run': analytic['n_total'],
            'tests_passed': analytic['n_passed'],
            'all_passed': analytic['all_passed'],
            'max_residual': analytic['max_residual_overall'],
            'max_disagreement': analytic['max_disagreement_overall']
        }
        
        # Table: Analytic validation results
        rows = []
        for r in analytic['individual_results']:
            rows.append({
                'test': r['test_name'],
                'description': r['description'],
                'max_residual': f"{r['max_residual']:.2e}",
                'max_disagreement': f"{r['max_disagreement']:.2e}",
                'status': 'PASS' if r['passed'] else 'FAIL'
            })
        manuscript_data['tables']['analytic_validation'] = rows
    
    # QuSpin validation summary
    if all_results.get('quspin', {}).get('status') == 'success':
        quspin = all_results['quspin']['result']
        manuscript_data['summary']['quspin'] = {
            'available': quspin.get('quspin_available', False),
            'tests_run': quspin.get('n_total', 0),
            'tests_passed': quspin.get('n_passed', 0)
        }
        
        if quspin.get('quspin_available'):
            for r in quspin.get('individual_results', []):
                if 'max_absolute_difference' in r:
                    manuscript_data['key_numbers']['quspin_max_abs_diff'] = r['max_absolute_difference']
                    manuscript_data['key_numbers']['quspin_mean_abs_diff'] = r.get('mean_absolute_difference', 0)
    
    # Performance scaling
    if all_results.get('performance', {}).get('status') == 'success':
        perf = all_results['performance']['result']
        scaling = perf['individual_results'][0]
        
        rows = []
        for s in scaling['sizes']:
            rows.append({
                'N': s['n_sites'],
                'd': s['dimension'],
                'build_ms': f"{s['build_time']['mean_ms']:.2f} ± {s['build_time']['std_ms']:.2f}",
                'single_ms': f"{s['single_oracle_time']['mean_ms']:.2f} ± {s['single_oracle_time']['std_ms']:.2f}",
                'multi_ms': f"{s['multi_oracle_time']['mean_ms']:.2f} ± {s['multi_oracle_time']['std_ms']:.2f}",
                'overhead': f"{s['consensus_overhead']:.2f}x"
            })
        manuscript_data['tables']['performance_scaling'] = rows
        
        # Average overhead
        overheads = [s['consensus_overhead'] for s in scaling['sizes']]
        manuscript_data['key_numbers']['mean_consensus_overhead'] = sum(overheads) / len(overheads)
    
    # Platform info
    if all_results.get('platform', {}).get('status') == 'success':
        platform_data = all_results['platform']['result']
        manuscript_data['summary']['platform'] = platform_data['platform_info']
    
    # Error injection
    if all_results.get('error_injection', {}).get('status') == 'success':
        ei = all_results['error_injection']['result']
        manuscript_data['summary']['error_injection'] = {
            'tests_run': ei['n_total'],
            'tests_passed': ei['n_passed'],
            'all_detected': ei['all_passed']
        }
    
    return manuscript_data


def create_archive(results_dir: Path, archive_path: Path) -> None:
    """Create tar.gz archive of all results."""
    print_section(f"Creating archive: {archive_path.name}")
    
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(str(results_dir), arcname=results_dir.name)
    
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"    Archive size: {size_mb:.2f} MB")
    print(f"    Location: {archive_path}")


def main():
    parser = argparse.ArgumentParser(
        description='CERTIFY-ED Master Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmarks (smaller sizes, fewer runs)')
    parser.add_argument('--no-tests', action='store_true',
                        help='Skip pytest test suite')
    parser.add_argument('--no-quspin', action='store_true',
                        help='Skip QuSpin validation (force skip)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: results/)')
    args = parser.parse_args()
    
    print_header("CERTIFY-ED MASTER BENCHMARK RUNNER", "=")
    print(f"\nStart time: {datetime.now().isoformat()}")
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    
    # Setup output directory
    if args.output_dir:
        results_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = SCRIPT_DIR / "results" / f"run_{timestamp}"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {results_dir}")
    
    overall_t0 = time.time()
    all_results = {}
    
    # 1. Platform info (always run first)
    print_header("PHASE 1: Platform Information")
    all_results['platform'] = safe_run(
        "Platform Benchmarks", run_platform_benchmarks
    )
    if all_results['platform']['status'] == 'success':
        with open(results_dir / "platform_results.json", 'w') as f:
            json.dump(all_results['platform']['result'], f, indent=2, default=str)
    
    # 2. Test suite
    if not args.no_tests:
        print_header("PHASE 2: Test Suite")
        test_result = run_pytest(results_dir)
        all_results['tests'] = {'status': 'success', 'result': test_result}
        with open(results_dir / "test_results.json", 'w') as f:
            json.dump(test_result, f, indent=2)
    
    # 3. Analytic benchmarks
    print_header("PHASE 3: Analytic Validation")
    all_results['analytic'] = safe_run(
        "Analytic Benchmarks", run_analytic_benchmarks
    )
    if all_results['analytic']['status'] == 'success':
        with open(results_dir / "analytic_results.json", 'w') as f:
            json.dump(all_results['analytic']['result'], f, indent=2, default=str)
    
    # 4. QuSpin validation (optional)
    if not args.no_quspin:
        print_header("PHASE 4: QuSpin Cross-Validation")
        all_results['quspin'] = safe_run(
            "QuSpin Validation", run_quspin_validation
        )
        if all_results['quspin']['status'] == 'success':
            with open(results_dir / "quspin_results.json", 'w') as f:
                json.dump(all_results['quspin']['result'], f, indent=2, default=str)
    
    # 5. Error injection tests
    print_header("PHASE 5: Error Injection Tests")
    all_results['error_injection'] = safe_run(
        "Error Injection Tests", run_error_injection_tests
    )
    if all_results['error_injection']['status'] == 'success':
        with open(results_dir / "error_injection_results.json", 'w') as f:
            json.dump(all_results['error_injection']['result'], f, indent=2, default=str)
    
    # 6. Performance benchmarks (most time consuming)
    print_header("PHASE 6: Performance Benchmarks")
    all_results['performance'] = safe_run(
        "Performance Benchmarks", run_performance_benchmarks, args.quick
    )
    if all_results['performance']['status'] == 'success':
        with open(results_dir / "performance_results.json", 'w') as f:
            json.dump(all_results['performance']['result'], f, indent=2, default=str)
    
    # 7. Generate figures
    print_header("PHASE 7: Generate Figures")
    figure_info = safe_run(
        "Figure Generation", generate_figures, results_dir, all_results
    )
    
    # 8. Aggregate manuscript data
    print_header("PHASE 8: Manuscript Data")
    manuscript_data = create_manuscript_data(all_results)
    with open(results_dir / "manuscript_data.json", 'w') as f:
        json.dump(manuscript_data, f, indent=2, default=str)
    print("    manuscript_data.json created with aggregated key numbers and tables")
    
    # 9. Create manifest
    total_elapsed = time.time() - overall_t0
    
    manifest = {
        'run_timestamp': datetime.now().isoformat(),
        'total_elapsed_seconds': total_elapsed,
        'mode': 'quick' if args.quick else 'full',
        'flags': {
            'no_tests': args.no_tests,
            'no_quspin': args.no_quspin,
        },
        'phases': {
            name: {
                'status': res.get('status', 'unknown'),
                'elapsed_seconds': res.get('elapsed_seconds', 0)
            }
            for name, res in all_results.items()
        },
        'output_files': sorted([f.name for f in results_dir.iterdir() if f.is_file()]),
        'output_directories': sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    }
    
    with open(results_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    # 10. Create archive
    print_header("PHASE 9: Archive Creation")
    archive_name = f"certify_ed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    archive_path = SCRIPT_DIR / archive_name
    create_archive(results_dir, archive_path)
    
    # Final summary
    print_header("SUMMARY", "=")
    print(f"Total runtime: {total_elapsed:.1f} seconds")
    print(f"\nPhase Results:")
    for name, res in all_results.items():
        status = res.get('status', 'unknown').upper()
        elapsed = res.get('elapsed_seconds', 0)
        marker = "OK" if status == 'SUCCESS' else "X"
        print(f"  [{marker}] {name:25s} {status:10s} ({elapsed:.1f}s)")
    
    print(f"\nOutput directory: {results_dir}")
    print(f"Archive:          {archive_path}")
    print(f"\n=> Send {archive_path.name} for manuscript revision")
    print()
    
    # Return non-zero if anything failed
    has_failures = any(
        res.get('status') == 'failed' for res in all_results.values()
    )
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
