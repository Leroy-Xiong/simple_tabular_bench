"""
Visualization utilities for benchmark results.
Creates plots comparing tree-based and deep learning models.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(filename: str = 'multi_dataset_results.json'):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_model_comparison(results, metric='roc_auc', save_path=None):
    """
    Plot model performance comparison across datasets.
    
    Args:
        results: List of result dictionaries
        metric: Metric to plot ('roc_auc', 'accuracy', or 'rmse')
        save_path: Path to save figure
    """
    # Prepare data
    data = []
    for dataset_result in results:
        dataset_name = dataset_result['dataset']
        for r in dataset_result['results']:
            data.append({
                'Dataset': dataset_name,
                'Model': r['model'],
                'Type': r['type'],
                'ROC-AUC': r.get('roc_auc', 0),
                'Accuracy': r.get('accuracy', 0),
                'Train Time (s)': r['train_time']
            })
    
    df = pd.DataFrame(data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Performance comparison
    ax1 = axes[0]
    pivot_df = df.pivot(index='Dataset', columns='Model', values='ROC-AUC')
    pivot_df.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Model Performance Comparison (ROC-AUC)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ROC-AUC', fontsize=10)
    ax1.set_xlabel('Dataset', fontsize=10)
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Plot 2: Tree vs Deep Learning
    ax2 = axes[1]
    type_comparison = df.groupby(['Dataset', 'Type'])['ROC-AUC'].max().unstack()
    type_comparison.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'], width=0.6)
    ax2.set_title('Tree-based vs Deep Learning (Best Model)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROC-AUC', fontsize=10)
    ax2.set_xlabel('Dataset', fontsize=10)
    ax2.legend(title='Model Type')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def plot_training_time(results, save_path=None):
    """
    Plot training time comparison.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save figure
    """
    # Prepare data
    data = []
    for dataset_result in results:
        dataset_name = dataset_result['dataset']
        for r in dataset_result['results']:
            data.append({
                'Dataset': dataset_name,
                'Model': r['model'],
                'Type': r['type'],
                'Train Time (s)': r['train_time']
            })
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by model type and calculate average
    pivot_df = df.pivot(index='Dataset', columns='Model', values='Train Time (s)')
    
    # Use log scale for better visualization
    pivot_df.plot(kind='bar', ax=ax, logy=True)
    ax.set_title('Training Time Comparison (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds, log scale)', fontsize=10)
    ax.set_xlabel('Dataset', fontsize=10)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def plot_model_ranking(results, save_path=None):
    """
    Plot average model ranking across datasets.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save figure
    """
    # Prepare data
    model_scores = {}
    model_types = {}
    
    for dataset_result in results:
        # Sort models by ROC-AUC for this dataset
        sorted_models = sorted(dataset_result['results'], 
                              key=lambda x: x.get('roc_auc', 0), 
                              reverse=True)
        
        for rank, r in enumerate(sorted_models, 1):
            model = r['model']
            if model not in model_scores:
                model_scores[model] = []
                model_types[model] = r['type']
            model_scores[model].append(rank)
    
    # Calculate average rank
    avg_ranks = {model: np.mean(ranks) for model, ranks in model_scores.items()}
    
    # Sort by average rank
    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [m[0] for m in sorted_models]
    ranks = [m[1] for m in sorted_models]
    colors = ['#2ecc71' if model_types[m] == 'tree-based' else '#e74c3c' for m in models]
    
    bars = ax.barh(models, ranks, color=colors)
    ax.set_xlabel('Average Rank (lower is better)', fontsize=10)
    ax.set_title('Model Ranking Across All Datasets', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{rank:.2f}', va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Tree-based'),
        Patch(facecolor='#e74c3c', label='Deep Learning')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def plot_performance_vs_time(results, save_path=None):
    """
    Plot performance vs training time trade-off.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, dataset_result in enumerate(results):
        ax = axes[idx]
        dataset_name = dataset_result['dataset']
        
        for r in dataset_result['results']:
            color = '#2ecc71' if r['type'] == 'tree-based' else '#e74c3c'
            marker = 'o' if r['type'] == 'tree-based' else 's'
            ax.scatter(r['train_time'], r.get('roc_auc', 0), 
                      s=200, c=color, marker=marker, alpha=0.7,
                      edgecolors='black', linewidth=1)
            ax.annotate(r['model'], (r['train_time'], r.get('roc_auc', 0)),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Training Time (seconds)', fontsize=10)
        ax.set_ylabel('ROC-AUC', fontsize=10)
        ax.set_title(f'{dataset_name}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    # Add overall legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', 
               markersize=10, label='Tree-based', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', 
               markersize=10, label='Deep Learning', markeredgecolor='black')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.suptitle('Performance vs Training Time Trade-off', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def create_summary_table(results):
    """
    Create a summary table of results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        pandas DataFrame with summary statistics
    """
    summary_data = []
    
    for dataset_result in results:
        dataset_name = dataset_result['dataset']
        task_type = dataset_result['task_type']
        n_samples = dataset_result['n_samples_train'] + dataset_result['n_samples_test']
        n_features = dataset_result['n_features']
        
        # Find best models
        best_tree = max([r for r in dataset_result['results'] if r['type'] == 'tree-based'],
                       key=lambda x: x.get('roc_auc', 0))
        best_deep = max([r for r in dataset_result['results'] if r['type'] == 'deep-learning'],
                       key=lambda x: x.get('roc_auc', 0))
        
        summary_data.append({
            'Dataset': dataset_name,
            'Samples': n_samples,
            'Features': n_features,
            'Best Tree Model': best_tree['model'],
            'Tree ROC-AUC': f"{best_tree.get('roc_auc', 0):.4f}",
            'Tree Time (s)': f"{best_tree['train_time']:.2f}",
            'Best Deep Model': best_deep['model'],
            'Deep ROC-AUC': f"{best_deep.get('roc_auc', 0):.4f}",
            'Deep Time (s)': f"{best_deep['train_time']:.2f}",
            'Gap': f"{best_tree.get('roc_auc', 0) - best_deep.get('roc_auc', 0):.4f}"
        })
    
    df = pd.DataFrame(summary_data)
    return df


def main():
    """Main function to generate all visualizations."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Load results
    print("Loading results...")
    results = load_results('multi_dataset_results.json')
    
    # Create output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    print("\n1. Model comparison plot...")
    plot_model_comparison(results, save_path=output_dir / 'model_comparison.png')
    
    print("\n2. Training time comparison...")
    plot_training_time(results, save_path=output_dir / 'training_time.png')
    
    print("\n3. Model ranking...")
    plot_model_ranking(results, save_path=output_dir / 'model_ranking.png')
    
    print("\n4. Performance vs time trade-off...")
    plot_performance_vs_time(results, save_path=output_dir / 'performance_vs_time.png')
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    summary_df = create_summary_table(results)
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"All figures saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
