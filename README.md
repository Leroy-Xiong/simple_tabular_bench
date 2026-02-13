# Tabular Data Benchmark

Reproduction of experiments from "Why do tree-based models still outperform deep learning on typical tabular data?" (NeurIPS 2022)

## Quick Start

```bash
# Create conda environment
conda create -n tabular_benchmark python=3.10 -y
conda activate tabular_benchmark

# Install dependencies
pip install -r requirements.txt

# Run benchmark on 3 datasets
python run_simple_benchmark.py

# Generate visualizations
python visualize_results.py
```

## Project Structure

```
├── data_loader.py              # OpenML dataset loading
├── benchmark.py                # Main benchmark class
├── run_simple_benchmark.py     # Run experiments on 3 datasets
├── visualize_results.py        # Result visualization
├── models/
│   ├── tree_models.py          # RandomForest, XGBoost, LightGBM
│   └── deep_models.py          # MLP, TabNet
└── figures/                    # Generated plots
```

## Models

**Tree-based:**
- Random Forest
- XGBoost
- LightGBM

**Deep Learning:**
- MLP (Multi-Layer Perceptron)
- TabNet

## Datasets

Default 3 datasets from OpenML:
- `credit-g`: German Credit (1,000 samples, 20 features)
- `diabetes`: Pima Indians Diabetes (768 samples, 8 features)
- `credit`: Credit Default (16,714 samples, 10 features)

Modify `DATASETS` in `run_simple_benchmark.py` to add more.

## Results

Results are saved to `multi_dataset_results.json`.

Example output shows tree models consistently outperform deep learning on tabular data with 10-100x faster training.

## Customization

Add new models in `models/`, then update `benchmark.py` to include them in comparisons.
