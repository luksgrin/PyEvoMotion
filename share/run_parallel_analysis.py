#!/usr/bin/env python3
"""
Parallel execution script for running multiple PyEvoMotion analyses.

This script runs the UK and USA analyses multiple times to assess
reproducibility and variance due to nonlinear fitting randomness.
Results are saved in separate subdirectories for each run.

Usage:
    python run_parallel_analysis.py [batch_name] [num_runs] [max_workers]
    
Examples:
    python run_parallel_analysis.py batch2           # Run batch2 with 5 runs
    python run_parallel_analysis.py batch3 10        # Run batch3 with 10 runs
    python run_parallel_analysis.py batch4 5 4       # Run batch4 with 5 runs, max 4 parallel workers
"""

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_analysis(country: str, run_number: int, base_output_dir: Path) -> tuple[str, int, bool]:
    """
    Run a single PyEvoMotion analysis for UK or USA dataset.
    
    :param country: Either "UK" or "USA"
    :type country: str
    :param run_number: The run number for this batch
    :type run_number: int
    :param base_output_dir: Base directory for output files
    :type base_output_dir: Path
    :return: Tuple of (country, run_number, success_status)
    :rtype: tuple[str, int, bool]
    """
    # Create output directory for this specific run
    output_dir = base_output_dir / f"{country}_run{run_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file prefix (without extension - PyEvoMotion will add suffixes)
    output_prefix = output_dir / f"fig{country}"
    
    # Build the command
    cmd = [
        "PyEvoMotion",
        f"tests/data/test3/test3{country}.fasta",
        f"share/figdata{country}.tsv",
        str(output_prefix),
        "-k", "total",
        "-dt", "7D",
        "-dr", "2020-10-01..2021-08-01",
        "-ep",
        "-xj",
    ]
    
    print(f"Starting {country} run {run_number}...")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print(f"Completed {country} run {run_number}")
        
        # Optionally save stdout/stderr to log files
        log_file = output_dir / "run.log"
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        
        return (country, run_number, True)
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {country} run {run_number}: {e}")
        
        # Save error log
        error_log = output_dir / "error.log"
        with open(error_log, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"\nReturn code: {e.returncode}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(e.stdout if e.stdout else "No stdout")
            f.write("\n=== STDERR ===\n")
            f.write(e.stderr if e.stderr else "No stderr")
        
        return (country, run_number, False)
    except Exception as e:
        print(f"UNEXPECTED ERROR in {country} run {run_number}: {e}")
        return (country, run_number, False)


def main():
    """
    Main execution function for parallel batch analysis.
    
    Runs UK and USA PyEvoMotion analyses multiple times in parallel to assess
    reproducibility. Results are saved to batch subdirectories with configurable
    parallelism and run counts via command line arguments.
    """
    
    # Parse command line arguments
    batch_name = sys.argv[1] if len(sys.argv) > 1 else "batch1"
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    
    # Configuration
    NUM_RUNS = num_runs
    COUNTRIES = ["UK", "USA"]
    BASE_OUTPUT_DIR = Path(f"share/test-runs/{batch_name}")
    
    # Check if directory already exists and warn user
    if BASE_OUTPUT_DIR.exists():
        existing_files = list(BASE_OUTPUT_DIR.glob("*"))
        if existing_files:
            print(f"WARNING: {BASE_OUTPUT_DIR} already exists with {len(existing_files)} items!")
            response = input("Continue anyway? This may overwrite existing files. [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Create base output directory
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {BASE_OUTPUT_DIR}")
    print(f"Batch name: {batch_name}")
    
    # Generate all tasks (combinations of country and run number)
    tasks = [
        (country, run_num)
        for country in COUNTRIES
        for run_num in range(1, NUM_RUNS + 1)
    ]
    
    print(f"\nTotal number of tasks: {len(tasks)}")
    print(f"Running {NUM_RUNS} runs for each of {COUNTRIES}")
    print(f"Max parallel workers: {max_workers}")
    print(f"Output will be saved to: {BASE_OUTPUT_DIR}/")
    print("\nStarting parallel execution...\n")
    
    # Run tasks in parallel
    # Use max_workers to control parallelism (None = number of CPUs)
    # Adjust this if you want to limit parallel processes
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_analysis, country, run_num, BASE_OUTPUT_DIR): (country, run_num)
            for country, run_num in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            country, run_num, success = future.result()
            results.append((country, run_num, success))
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]
    
    print(f"\nTotal runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed runs:")
        for country, run_num, _ in failed:
            print(f"  - {country} run {run_num}")
    
    print(f"\nAll results saved to: {BASE_OUTPUT_DIR}/")
    print("\nDirectory structure:")
    for country in COUNTRIES:
        for run_num in range(1, NUM_RUNS + 1):
            run_dir = BASE_OUTPUT_DIR / f"{country}_run{run_num}"
            status = "✓" if (country, run_num, True) in results else "✗"
            print(f"  {status} {run_dir}/")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

