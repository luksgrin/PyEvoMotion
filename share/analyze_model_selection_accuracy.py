#!/usr/bin/env python3
"""
Script to analyze model selection accuracy from test5 regression results.

This script analyzes the out_regression_results.json files from both linear and powerlaw
test datasets to compute accuracy metrics and create visualizations.

Success criteria:
- Linear datasets: success when "selected" field is "linear"
- Powerlaw datasets: success when "selected" field is "power_law"
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_regression_results(directory: str) -> List[Dict]:
    """Load all regression results from a directory."""
    results = []
    pattern = os.path.join(directory, "**", "*out_regression_results.json")
    
    for file_path in glob.glob(pattern, recursive=True):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract the model selection info
                model_selection = data.get("scaled var number of substitutions model", {}).get("model_selection", {})
                results.append({
                    'file': file_path,
                    'selected_model': model_selection.get("selected", "unknown"),
                    'linear_AIC': model_selection.get("linear_AIC", None),
                    'power_law_AIC': model_selection.get("power_law_AIC", None),
                    'delta_AIC_linear': model_selection.get("delta_AIC_linear", None),
                    'delta_AIC_power_law': model_selection.get("delta_AIC_power_law", None),
                    'akaike_weight_linear': model_selection.get("akaike_weight_linear", None),
                    'akaike_weight_power_law': model_selection.get("akaike_weight_power_law", None)
                })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results


def analyze_model_selection_accuracy():
    """Analyze model selection accuracy and create visualizations."""
    
    # Define paths
    base_path = Path(__file__).parent.parent / "tests" / "data" / "test5"
    linear_dir = base_path / "linear" / "output"
    powerlaw_dir = base_path / "powerlaw" / "output"
    
    print("Loading regression results...")
    
    # Load results from both directories
    linear_results = load_regression_results(str(linear_dir))
    powerlaw_results = load_regression_results(str(powerlaw_dir))
    
    print(f"Loaded {len(linear_results)} linear results")
    print(f"Loaded {len(powerlaw_results)} powerlaw results")
    
    # Analyze linear dataset results
    linear_success = sum(1 for r in linear_results if r['selected_model'] == 'linear')
    linear_failure = len(linear_results) - linear_success
    
    # Analyze powerlaw dataset results  
    powerlaw_success = sum(1 for r in powerlaw_results if r['selected_model'] == 'power_law')
    powerlaw_failure = len(powerlaw_results) - powerlaw_success
    
    # Create summary table
    summary_data = {
        'Dataset Type': ['Linear', 'Powerlaw'],
        'Total Tests': [len(linear_results), len(powerlaw_results)],
        'Successes': [linear_success, powerlaw_success],
        'Failures': [linear_failure, powerlaw_failure],
        'Success Rate': [linear_success/len(linear_results) if linear_results else 0, 
                        powerlaw_success/len(powerlaw_results) if powerlaw_results else 0]
    }
    
    df = pd.DataFrame(summary_data)
    print("\nModel Selection Accuracy Summary:")
    print("=" * 50)
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Calculate overall accuracy metrics
    total_tests = len(linear_results) + len(powerlaw_results)
    total_successes = linear_success + powerlaw_success
    overall_accuracy = total_successes / total_tests if total_tests > 0 else 0
    
    # Calculate precision and recall for each model type
    # For linear: TP = linear_success, FP = powerlaw_failure, FN = linear_failure, TN = powerlaw_success
    linear_tp = linear_success
    linear_fp = powerlaw_failure  # Powerlaw datasets incorrectly classified as linear
    linear_fn = linear_failure    # Linear datasets incorrectly classified as powerlaw
    linear_tn = powerlaw_success  # Powerlaw datasets correctly classified as powerlaw
    
    # For powerlaw: TP = powerlaw_success, FP = linear_failure, FN = powerlaw_failure, TN = linear_success
    powerlaw_tp = powerlaw_success
    powerlaw_fp = linear_failure  # Linear datasets incorrectly classified as powerlaw
    powerlaw_fn = powerlaw_failure  # Powerlaw datasets incorrectly classified as linear
    powerlaw_tn = linear_success  # Linear datasets correctly classified as linear
    
    # Calculate metrics
    linear_precision = linear_tp / (linear_tp + linear_fp) if (linear_tp + linear_fp) > 0 else 0
    linear_recall = linear_tp / (linear_tp + linear_fn) if (linear_tp + linear_fn) > 0 else 0
    linear_specificity = linear_tn / (linear_tn + linear_fp) if (linear_tn + linear_fp) > 0 else 0
    
    powerlaw_precision = powerlaw_tp / (powerlaw_tp + powerlaw_fp) if (powerlaw_tp + powerlaw_fp) > 0 else 0
    powerlaw_recall = powerlaw_tp / (powerlaw_tp + powerlaw_fn) if (powerlaw_tp + powerlaw_fn) > 0 else 0
    powerlaw_specificity = powerlaw_tn / (powerlaw_tn + powerlaw_fp) if (powerlaw_tn + powerlaw_fp) > 0 else 0
    
    # F1 scores
    linear_f1 = 2 * (linear_precision * linear_recall) / (linear_precision + linear_recall) if (linear_precision + linear_recall) > 0 else 0
    powerlaw_f1 = 2 * (powerlaw_precision * powerlaw_recall) / (powerlaw_precision + powerlaw_recall) if (powerlaw_precision + powerlaw_recall) > 0 else 0
    
    print(f"\nOverall Accuracy: {overall_accuracy:.3f} ({total_successes}/{total_tests})")
    print("\nDetailed Metrics:")
    print("=" * 50)
    
    metrics_data = {
        'Model Type': ['Linear', 'Powerlaw'],
        'Precision': [linear_precision, powerlaw_precision],
        'Recall (Sensitivity)': [linear_recall, powerlaw_recall],
        'Specificity': [linear_specificity, powerlaw_specificity],
        'F1-Score': [linear_f1, powerlaw_f1]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False, float_format='%.3f'))
    
    # Create confusion matrix data
    confusion_matrix = np.array([
        [linear_tp, linear_fp],    # True Linear, False Linear
        [linear_fn, linear_tn]     # False Powerlaw, True Powerlaw
    ])
    
    print(f"\nConfusion Matrix:")
    print("=" * 30)
    print("                Predicted")
    print("              Linear  Powerlaw")
    print(f"Actual Linear   {linear_tp:3d}     {linear_fp:3d}")
    print(f"       Powerlaw {linear_fn:3d}     {linear_tn:3d}")
    
    # Create visualizations
    create_bar_chart(summary_data, overall_accuracy)
    create_confusion_matrix_heatmap(confusion_matrix)
    create_metrics_comparison(metrics_data)
    
    # Save detailed results
    save_detailed_results(linear_results, powerlaw_results, summary_data, metrics_data, overall_accuracy)
    
    return df, metrics_df, overall_accuracy


def create_bar_chart(summary_data: Dict, overall_accuracy: float):
    """Create a bar chart showing success rates."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart for success/failure counts
    x = np.arange(len(summary_data['Dataset Type']))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, summary_data['Successes'], width, label='Successes', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, summary_data['Failures'], width, label='Failures', color='red', alpha=0.7)
    
    ax1.set_xlabel('Dataset Type')
    ax1.set_ylabel('Number of Tests')
    ax1.set_title('Model Selection Results by Dataset Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_data['Dataset Type'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Success rate bar chart
    bars3 = ax2.bar(summary_data['Dataset Type'], summary_data['Success Rate'], 
                   color=['blue', 'orange'], alpha=0.7)
    
    # Add overall accuracy line
    ax2.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Accuracy: {overall_accuracy:.3f}')
    
    ax2.set_xlabel('Dataset Type')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Model Selection Success Rates')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_selection_accuracy_chart.pdf', dpi=300, bbox_inches='tight')


def create_confusion_matrix_heatmap(confusion_matrix: np.ndarray):
    """Create a heatmap of the confusion matrix."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Linear', 'Powerlaw'])
    ax.set_yticklabels(['Linear', 'Powerlaw'])
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix: Model Selection Results')
    
    plt.tight_layout()
    plt.savefig('share/confusion_matrix_heatmap.pdf', dpi=300, bbox_inches='tight')


def create_metrics_comparison(metrics_data: Dict):
    """Create a comparison chart of different metrics."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics_data['Model Type']))
    width = 0.2
    
    metrics = ['Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = metrics_data[metric]
        ax.bar(x + i * width, values, width, label=metric, color=color, alpha=0.7)
    
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Score')
    ax.set_title('Model Selection Performance Metrics Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_data['Model Type'])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        values = metrics_data[metric]
        for j, value in enumerate(values):
            ax.text(j + i * width, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('share/metrics_comparison_chart.pdf', dpi=300, bbox_inches='tight')


def save_detailed_results(linear_results: List[Dict], powerlaw_results: List[Dict], 
                         summary_data: Dict, metrics_data: Dict, overall_accuracy: float):
    """Save detailed results to JSON file."""
    
    results = {
        'overall_accuracy': overall_accuracy,
        'summary': summary_data,
        'metrics': metrics_data,
        'linear_results': linear_results,
        'powerlaw_results': powerlaw_results,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('model_selection_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved as 'model_selection_analysis_results.json'")


if __name__ == "__main__":
    print("Model Selection Accuracy Analysis")
    print("=" * 40)
    print("Analyzing regression results from test5 datasets...")
    print("Success criteria:")
    print("- Linear datasets: success when 'selected' = 'linear'")
    print("- Powerlaw datasets: success when 'selected' = 'power_law'")
    print()
    
    try:
        summary_df, metrics_df, accuracy = analyze_model_selection_accuracy()
        print(f"\nAnalysis complete! Overall accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
