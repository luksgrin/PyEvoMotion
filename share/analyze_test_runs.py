#!/usr/bin/env python3
"""
Analyze parameter variability across multiple PyEvoMotion runs.

This script loads regression results from multiple runs and creates
violin plots to visualize parameter distributions and assess
reproducibility of the nonlinear fitting process.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def set_matplotlib_params():
    """Set consistent matplotlib styling."""
    mpl_params = {
        "font.sans-serif": "Helvetica",
        "axes.linewidth": 2,
        "axes.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 12,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.frameon": False,
    }
    for k, v in mpl_params.items():
        mpl.rcParams[k] = v


def load_regression_results(base_dir: Path, country: str, num_runs: int = 5) -> List[Dict]:
    """
    Load regression results from multiple runs.
    
    Args:
        base_dir: Base directory containing run subdirectories
        country: Either "UK" or "USA"
        num_runs: Number of runs to load
        
    Returns:
        List of dictionaries containing regression results
    """
    results = []
    
    for run_num in range(1, num_runs + 1):
        run_dir = base_dir / f"{country}_run{run_num}"
        results_file = run_dir / f"fig{country}_regression_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'run': run_num,
                    'country': country,
                    'data': data
                })
        else:
            print(f"Warning: {results_file} not found")
    
    return results


def extract_parameters(results: List[Dict]) -> pd.DataFrame:
    """
    Extract parameters from regression results into a DataFrame.
    
    Args:
        results: List of regression result dictionaries
        
    Returns:
        DataFrame with parameters from all runs
    """
    records = []
    
    for result in results:
        run = result['run']
        country = result['country']
        data = result['data']
        
        record = {
            'run': run,
            'country': country
        }
        
        # Extract mean model parameters
        mean_key = None
        for key in ["mean number of mutations model", 
                    "mean number of mutations per 7D model",
                    "mean number of substitutions model"]:
            if key in data:
                mean_key = key
                break
        
        if mean_key:
            mean_model = data[mean_key]
            record['mean_m'] = mean_model['parameters']['m']
            record['mean_b'] = mean_model['parameters']['b']
            record['mean_r2'] = mean_model['r2']
        
        # Extract variance model parameters
        var_key = None
        for key in ["scaled var number of mutations model",
                    "scaled var number of mutations per 7D model",
                    "scaled var number of substitutions model"]:
            if key in data:
                var_key = key
                break
        
        if var_key:
            var_model = data[var_key]
            
            # Check if model selection was performed
            if "model_selection" in var_model:
                selected = var_model["model_selection"]["selected"]
                record['var_model_selected'] = selected
                
                if selected == "linear" and "linear_model" in var_model:
                    linear = var_model["linear_model"]
                    record['var_m'] = linear['parameters']['m']
                    record['var_r2'] = linear['r2']
                    record['var_d'] = None
                    record['var_alpha'] = None
                    
                elif selected == "power_law" and "power_law_model" in var_model:
                    power_law = var_model["power_law_model"]
                    record['var_d'] = power_law['parameters']['d']
                    record['var_alpha'] = power_law['parameters']['alpha']
                    record['var_r2'] = power_law['r2']
                    record['var_m'] = None
            else:
                # Old format without model selection
                params = var_model['parameters']
                record['var_r2'] = var_model['r2']
                
                if 'm' in params:
                    record['var_m'] = params['m']
                    record['var_d'] = None
                    record['var_alpha'] = None
                    record['var_model_selected'] = 'linear'
                elif 'd' in params and 'alpha' in params:
                    record['var_d'] = params['d']
                    record['var_alpha'] = params['alpha']
                    record['var_m'] = None
                    record['var_model_selected'] = 'power_law'
        
        records.append(record)
    
    return pd.DataFrame(records)


def create_violin_plots(df: pd.DataFrame, export: bool = False, show: bool = True, output_filename: str = "share/test_runs_violin_plot.pdf"):
    """
    Create violin plots for parameter distributions.
    
    Args:
        df: DataFrame with extracted parameters
        export: Whether to save the figure
        show: Whether to display the figure
        output_filename: Path to save the figure (default: share/test_runs_violin_plot.pdf)
    """
    set_matplotlib_params()
    
    # Define colors
    colors = {
        "UK": "#76d6ff",
        "USA": "#FF6346",
    }
    
    # Parameters to plot
    mean_params = [
        ('mean_m', 'Mean: Slope (m)', 'mutations/week'),
        ('mean_b', 'Mean: Intercept (b)', 'mutations'),
        ('mean_r2', 'Mean: R²', '')
    ]
    
    # Check which variance model is predominantly used
    var_model_counts = df['var_model_selected'].value_counts()
    print("\nVariance model selection:")
    print(var_model_counts)
    
    # Determine which variance parameters to plot
    if var_model_counts.get('power_law', 0) > 0:
        var_params = [
            ('var_d', 'Variance: Coefficient (d)', ''),
            ('var_alpha', 'Variance: Exponent (α)', ''),
            ('var_r2', 'Variance: R²', '')
        ]
    else:
        var_params = [
            ('var_m', 'Variance: Slope (m)', 'mutations²/week'),
            ('var_r2', 'Variance: R²', '')
        ]
    
    all_params = mean_params + var_params
    
    # Create subplots
    n_params = len(all_params)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (param, title, unit) in enumerate(all_params):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Filter out None values for this parameter
        plot_df = df[df[param].notna()].copy()
        
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Create violin plot
        parts = ax.violinplot(
            [plot_df[plot_df['country'] == 'UK'][param].values,
             plot_df[plot_df['country'] == 'USA'][param].values],
            positions=[0, 1],
            showmeans=True,
            showextrema=True,
            widths=0.7
        )
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            country = ['UK', 'USA'][i]
            pc.set_facecolor(colors[country])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Style the other elements
        for partname in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(2)
        
        # Add scatter points for individual runs
        for i, country in enumerate(['UK', 'USA']):
            country_data = plot_df[plot_df['country'] == country]
            x_pos = np.random.normal(i, 0.04, size=len(country_data))
            ax.scatter(x_pos, country_data[param].values, 
                      alpha=0.6, s=50, c='black', zorder=3, edgecolors='white', linewidth=1)
        
        # Styling
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['UK', 'USA'])
        ax.set_ylabel(f'{title.split(": ")[1]} {f"({unit})" if unit else ""}'.strip())
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics text
        for i, country in enumerate(['UK', 'USA']):
            country_data = plot_df[plot_df['country'] == country][param]
            if len(country_data) > 0:
                mean_val = country_data.mean()
                std_val = country_data.std()
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                
                text_y = ax.get_ylim()[1] * 0.95 - i * (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
                ax.text(0.98, text_y, 
                       f'{country}: μ={mean_val:.4f}, σ={std_val:.4f}, CV={cv:.2f}%',
                       transform=ax.transData, ha='right', va='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor=colors[country], alpha=0.3))
    
    # Hide unused subplots
    for idx in range(len(all_params), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Parameter Variability Across Multiple Runs\n(Assessing Nonlinear Fitting Reproducibility)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if export:
        fig.savefig(output_filename, dpi=400, bbox_inches='tight')
        print(f"\nViolin plot saved as {output_filename}")
    
    if show:
        plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for all parameters."""
    print("\n" + "="*80)
    print("PARAMETER VARIABILITY SUMMARY")
    print("="*80)
    
    for country in ['UK', 'USA']:
        print(f"\n{country} Dataset:")
        print("-" * 40)
        
        country_df = df[df['country'] == country]
        
        # Mean model parameters
        print("\nMean Model:")
        for param in ['mean_m', 'mean_b', 'mean_r2']:
            if param in country_df.columns:
                values = country_df[param].dropna()
                if len(values) > 0:
                    mean = values.mean()
                    std = values.std()
                    cv = (std / mean * 100) if mean != 0 else 0
                    print(f"  {param:12s}: μ={mean:10.6f}, σ={std:10.6f}, CV={cv:6.2f}%")
        
        # Variance model parameters
        print("\nVariance Model:")
        var_model = country_df['var_model_selected'].mode()[0] if 'var_model_selected' in country_df.columns else 'unknown'
        print(f"  Selected model: {var_model}")
        
        if var_model == 'power_law':
            for param in ['var_d', 'var_alpha', 'var_r2']:
                if param in country_df.columns:
                    values = country_df[param].dropna()
                    if len(values) > 0:
                        mean = values.mean()
                        std = values.std()
                        cv = (std / mean * 100) if mean != 0 else 0
                        print(f"  {param:12s}: μ={mean:10.6f}, σ={std:10.6f}, CV={cv:6.2f}%")
        else:
            for param in ['var_m', 'var_r2']:
                if param in country_df.columns:
                    values = country_df[param].dropna()
                    if len(values) > 0:
                        mean = values.mean()
                        std = values.std()
                        cv = (std / mean * 100) if mean != 0 else 0
                        print(f"  {param:12s}: μ={mean:10.6f}, σ={std:10.6f}, CV={cv:6.2f}%")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        batch_name = sys.argv[1]
        BASE_DIR = Path(f"share/test-runs/{batch_name}")
        output_suffix = f"_{batch_name}"
    else:
        # Try to auto-detect batch directories or use batch1 as default
        test_runs_dir = Path("share/test-runs")
        batch_dirs = [d for d in test_runs_dir.iterdir() if d.is_dir() and d.name.startswith("batch")]
        
        if len(batch_dirs) == 0:
            # Fall back to old structure (no batch subdirectories)
            BASE_DIR = Path("share/test-runs")
            output_suffix = ""
        elif len(batch_dirs) == 1:
            # Use the only batch found
            BASE_DIR = batch_dirs[0]
            output_suffix = f"_{batch_dirs[0].name}"
            print(f"Auto-detected batch: {batch_dirs[0].name}")
        else:
            # Multiple batches - ask user or default to batch1
            print(f"Found {len(batch_dirs)} batches: {[d.name for d in batch_dirs]}")
            print("Please specify which batch to analyze:")
            print("  python analyze_test_runs.py batch1")
            print("Or analyze all batches separately by running for each.")
            return
    
    if not BASE_DIR.exists():
        print(f"Error: Directory {BASE_DIR} does not exist!")
        return
    
    # Auto-detect number of runs
    uk_runs = list(BASE_DIR.glob("UK_run*"))
    usa_runs = list(BASE_DIR.glob("USA_run*"))
    NUM_RUNS = max(len(uk_runs), len(usa_runs))
    
    COUNTRIES = ["UK", "USA"]
    
    print(f"Loading regression results from {BASE_DIR}...")
    print(f"Detected {NUM_RUNS} runs per country")
    
    # Load all results
    all_results = []
    for country in COUNTRIES:
        results = load_regression_results(BASE_DIR, country, NUM_RUNS)
        all_results.extend(results)
        print(f"Loaded {len(results)} runs for {country}")
    
    if not all_results:
        print("Error: No results found!")
        return
    
    # Extract parameters into DataFrame
    print("\nExtracting parameters...")
    df = extract_parameters(all_results)
    
    # Save to CSV for further analysis
    output_csv = f"share/test_runs_parameters{output_suffix}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Parameters saved to {output_csv}")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create violin plots
    print("\nCreating violin plots...")
    output_plot = f"share/test_runs_violin_plot{output_suffix}.pdf"
    create_violin_plots(df, export=True, show=True, output_filename=output_plot)


if __name__ == "__main__":
    main()

