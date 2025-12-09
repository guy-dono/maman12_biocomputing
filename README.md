# Bulk RNA-seq Deconvolution using Genetic Algorithm

A genetic algorithm implementation for deconvolving bulk RNA-seq data into cell type proportions.

## Problem Description

Given:
- **M**: Bulk expression matrix (genes x samples)
- **H**: Reference signature matrix (genes x celltypes)

Find **X** (celltypes x samples) such that **M = HX**, where each column of X represents cell type proportions for a sample (summing to 1).

## Installation

### Prerequisites
- Python >= 3.11
- Poetry or pip

### Using Poetry
```bash
poetry install
poetry run python main.py
```

### Using pip
```bash
pip install numpy pandas matplotlib scipy
python main.py
```