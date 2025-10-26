#!/usr/bin/env python3
"""
Parallel execution script for test5 synthetic dataset tests.

This script runs all linear and powerlaw synthetic dataset tests in parallel
to speed up execution time significantly.

Usage:
    python tests/run_test5_parallel.py [max_workers]
    
Examples:
    python tests/run_test5_parallel.py        # Use all CPU cores
    python tests/run_test5_parallel.py 10     # Max 10 parallel processes
"""

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


def run_single_test(dataset_type: str, dataset_num: str, timestamp: str) -> tuple[str, str, bool]:
    """
    Run a single PyEvoMotion test on a synthetic dataset.
    
    :param dataset_type: Either "linear" or "powerlaw"
    :type dataset_type: str
    :param dataset_num: Dataset number (formatted as "01", "02", etc.)
    :type dataset_num: str
    :param timestamp: Timestamp for organizing output directories
    :type timestamp: str
    :return: Tuple of (dataset_type, dataset_num, success_status)
    :rtype: tuple[str, str, bool]
    """
    
    # Paths
    base_path = f"tests/data/test5/{dataset_type}"
    input_fasta = f"{base_path}/synthdata_{dataset_type}_{dataset_num}.fasta"
    input_meta = f"{base_path}/synthdata_{dataset_type}_{dataset_num}.tsv"
    
    # Create subdirectory for each dataset within the timestamp directory
    output_dir = f"{base_path}/output/{timestamp}/{dataset_type}_{dataset_num}"
    output_prefix = f"{output_dir}/{dataset_type}_{dataset_num}_out"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "PyEvoMotion",
        input_fasta,
        input_meta,
        output_prefix,
        "-ep",
        "-k", "substitutions"
    ]
    
    print(f"Starting {dataset_type} dataset {dataset_num}...")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print(f"Completed {dataset_type} dataset {dataset_num}")
        
        # Save log
        log_file = f"{output_prefix}.log"
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        
        return (dataset_type, dataset_num, True)
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {dataset_type} dataset {dataset_num}: {e}")
        
        # Save error log
        error_log = f"{output_prefix}.error.log"
        with open(error_log, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"\nReturn code: {e.returncode}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(e.stdout if e.stdout else "No stdout")
            f.write("\n=== STDERR ===\n")
            f.write(e.stderr if e.stderr else "No stderr")
        
        return (dataset_type, dataset_num, False)
    except Exception as e:
        print(f"UNEXPECTED ERROR in {dataset_type} dataset {dataset_num}: {e}")
        return (dataset_type, dataset_num, False)


def main():
    """
    Main execution function for parallel test5 dataset analysis.
    
    Runs all linear and powerlaw synthetic dataset tests in parallel using PyEvoMotion.
    Results are saved to timestamped subdirectories for each dataset.
    Command line arguments allow control over the number of parallel workers.
    """
    
    # Parse command line arguments
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    
    # Generate timestamp for this batch
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Configuration
    NUM_DATASETS = 30
    DATASET_TYPES = ["linear", "powerlaw"]
    
    # Generate all tasks
    tasks = [
        (dataset_type, f"{i:02d}")
        for dataset_type in DATASET_TYPES
        for i in range(1, NUM_DATASETS + 1)
    ]
    
    print("="*60)
    print("TEST5 PARALLEL EXECUTION")
    print("="*60)
    print(f"\nTimestamp: {timestamp}")
    print(f"Total datasets: {len(tasks)}")
    print(f"  - Linear: {NUM_DATASETS}")
    print(f"  - Powerlaw: {NUM_DATASETS}")
    print(f"Max parallel workers: {max_workers}")
    print(f"\nOutput structure:")
    print(f"  tests/data/test5/linear/output/{timestamp}/")
    print(f"    ├── linear_01/")
    print(f"    ├── linear_02/")
    print(f"    └── ... (30 subdirectories)")
    print(f"  tests/data/test5/powerlaw/output/{timestamp}/")
    print(f"    ├── powerlaw_01/")
    print(f"    ├── powerlaw_02/")
    print(f"    └── ... (30 subdirectories)")
    print("\nStarting parallel execution...\n")
    print("="*60)
    
    # Run tasks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_test, dataset_type, dataset_num, timestamp): (dataset_type, dataset_num)
            for dataset_type, dataset_num in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            dataset_type, dataset_num, success = future.result()
            results.append((dataset_type, dataset_num, success))
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]
    
    linear_success = sum(1 for r in successful if r[0] == "linear")
    powerlaw_success = sum(1 for r in successful if r[0] == "powerlaw")
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"  - Linear: {linear_success}/{NUM_DATASETS}")
    print(f"  - Powerlaw: {powerlaw_success}/{NUM_DATASETS}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed tests:")
        for dataset_type, dataset_num, _ in failed:
            print(f"  - {dataset_type} dataset {dataset_num}")
    
    print(f"\nAll results saved to organized subdirectories:")
    print(f"  tests/data/test5/linear/output/{timestamp}/linear_{{01-30}}/")
    print(f"  tests/data/test5/powerlaw/output/{timestamp}/powerlaw_{{01-30}}/")
    
    print("\nNext steps:")
    print("  1. Analyze results with confusion matrix:")
    print("     python -c 'from share.manuscript_figure import create_confusion_matrix_plot; create_confusion_matrix_plot(export=True)'")
    print("  2. Or run full manuscript figure generation:")
    print("     python share/manuscript_figure.py")
    print("\nNote: The confusion matrix function will automatically find all results")
    print("      regardless of the directory structure (flat or nested).")
    
    print("\n" + "="*60)
    
    # Exit with appropriate code
    exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()

