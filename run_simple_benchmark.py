"""
Simple script to run benchmarks on multiple datasets.
This is a simplified version for quick experimentation.
"""

from benchmark import Benchmark
import json


# Define 3 datasets to benchmark
# Each dataset is defined by either name or (suite_id, task_id)
# Using well-known OpenML datasets from the paper's benchmark
DATASETS = [
    {'name': 'credit-g', 'description': 'German Credit (Classification)'},
    {'name': 'diabetes', 'description': 'Pima Indians Diabetes (Classification)'},
    {'suite_id': 337, 'task_id': None, 'description': 'Credit Default (Classification)'}
]


def run_single_dataset(dataset_config, run_tree=True, run_deep=True):
    """
    Run benchmark on a single dataset.
    
    Args:
        dataset_config: Dictionary with dataset configuration
        run_tree: Whether to run tree-based models
        run_deep: Whether to run deep learning models
    
    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_config.get('description', 'Unknown')}")
    print("=" * 70)
    
    # Create benchmark
    if 'name' in dataset_config:
        benchmark = Benchmark(dataset_name=dataset_config['name'])
    else:
        benchmark = Benchmark(
            suite_id=dataset_config.get('suite_id', 337),
            task_id=dataset_config.get('task_id')
        )
    
    # Run all models
    results_df = benchmark.run_all(run_tree=run_tree, run_deep=run_deep)
    
    # Print summary
    benchmark.print_summary()
    
    # Return results
    return {
        'dataset': dataset_config.get('name') or f"suite_{dataset_config.get('suite_id')}_task_{benchmark.task_id}",
        'description': dataset_config.get('description', ''),
        'task_type': benchmark.task_type,
        'n_samples_train': len(benchmark.X_train),
        'n_samples_test': len(benchmark.X_test),
        'n_features': len(benchmark.feature_names),
        'results': results_df.to_dict('records')
    }


def main():
    """
    Run benchmarks on multiple datasets and compare results.
    """
    print("=" * 70)
    print("MULTI-DATASET TABULAR DATA BENCHMARK")
    print("Based on: 'Why do tree-based models still outperform deep learning")
    print("           on typical tabular data?' (NeurIPS 2022)")
    print("=" * 70)
    
    all_results = []
    
    # Run benchmark on each dataset
    for i, dataset_config in enumerate(DATASETS, 1):
        print(f"\n\n>>> Running dataset {i}/{len(DATASETS)}")
        
        try:
            result = run_single_dataset(dataset_config, run_tree=True, run_deep=True)
            all_results.append(result)
        except Exception as e:
            print(f"Error running dataset {dataset_config}: {e}")
            continue
    
    # Print overall summary
    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY ACROSS ALL DATASETS")
    print("=" * 70)
    
    for result in all_results:
        print(f"\n--- {result['dataset']} ({result['description']}) ---")
        print(f"Samples: {result['n_samples_train']} train, {result['n_samples_test']} test")
        print(f"Features: {result['n_features']}")
        
        # Find best models
        df_results = result['results']
        
        # Get metric column
        if result['task_type'] == 'classification':
            metric_key = 'roc_auc' if any('roc_auc' in r for r in df_results) else 'accuracy'
        else:
            metric_key = 'r2' if any('r2' in r for r in df_results) else 'rmse'
        
        # Sort by metric
        sorted_results = sorted(df_results, key=lambda x: x.get(metric_key, 0), 
                               reverse=(metric_key != 'rmse'))
        
        print(f"\nTop 3 models by {metric_key}:")
        for j, r in enumerate(sorted_results[:3], 1):
            print(f"  {j}. {r['model']} ({r['type']}): {metric_key}={r.get(metric_key, 'N/A'):.4f}, "
                  f"train_time={r['train_time']:.2f}s")
    
    # Save all results
    with open('multi_dataset_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n\n" + "=" * 70)
    print("All results saved to multi_dataset_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
