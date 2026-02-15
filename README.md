# Tabular Data Benchmark

Reproduction of experiments from ["Why do tree-based models still outperform deep learning on typical tabular data?"](https://github.com/LeoGrin/tabular-benchmark) (NeurIPS 2022)

## Quick Start

### 1. Create and Activate Conda Environment

```bash
conda create -n tabular_benchmark python=3.13
conda activate tabular_benchmark
```

### 2. Install PyTorch

Visit [PyTorch Official Website](https://pytorch.org/get-started/locally/) to select the appropriate version for your CUDA version.

### 3. Install Other Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm xgboost lightgbm pytorch-tabnet openml
```

### 4. Run Benchmark

```bash
# Run benchmark on all datasets
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

Default 5 datasets from OpenML:

**Classification:**
- `credit-g`: German Credit (1,000 samples, 20 features)
- `diabetes`: Pima Indians Diabetes (768 samples, 8 features)
- `credit`: Credit Default (16,714 samples, 10 features)

**Regression:**
- `boston`: Boston Housing
- `fried`: Friedman

Modify `DATASETS` in `run_simple_benchmark.py` to add more.

## Results

Results are saved to `multi_dataset_results.json`.

Example output shows tree models consistently outperform deep learning on tabular data with 10-100x faster training.

## Customization

Add new models in `models/`, then update `benchmark.py` to include them in comparisons.
