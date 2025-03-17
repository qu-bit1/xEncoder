# xEncoder: Multi-Encoder Autoencoder for Xenium Datasets

## Problem Statement

This project addresses a key challenge in spatial transcriptomics: predicting gene expression for a larger set of genes using only a smaller set of genes. In spatial transcriptomics technologies like Xenium, there are often limitations on how many genes can be profiled simultaneously. This model enables researchers to:

1. Measure a smaller, carefully selected panel of genes
2. Computationally predict the expression of thousands of additional genes
3. Gain insights into cellular states and functions that would otherwise require more expensive or technically challenging experiments

## Alignment

First align both the tissues using the code given in the directory `tissue-alignment`

## Model Architecture

The project implements a multi-encoder single latent space single decoder architecture:

- **Source Encoder**: Processes the smaller gene set (input)
- **Target Encoder**: Processes the larger gene set (only used during training)
- **Shared Latent Space**: A common representation space that aligns both encoders
- **Decoder**: Reconstructs the larger gene set from the latent representation

Key features of the architecture:
- Shared initial transformation layer between encoders
- Batch normalization and dropout for regularization
- Cosine embedding loss to ensure latent space alignment
- Smooth L1 loss for reconstruction to handle outliers
- Softplus activation in the decoder to ensure non-negative gene expression

During training, both encoders are used to ensure alignment of the latent space. During inference, only the source encoder is used to predict the larger gene set.

## Data

The model works with two Xenium datasets:

1. **Source Dataset**: Contains a smaller set of genes (e.g., ~300 genes)
2. **Target Dataset**: Contains a larger set of genes (e.g., ~5000 genes)
3. **Alignment Information**: Maps cells between the two datasets

The data processing pipeline:
- Filters cells to include only those present in both datasets
- Ensures cells are properly aligned between datasets
- Normalizes gene expression values
- Identifies common genes between datasets for validation

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

To prepare the data for training:

```bash
python src/main.py --prepare_data
```

This will:
1. Copy raw data files (`cell_feature_matrix.h5`, `cell_feature_matrix2.h5`, `merged_aligned_cells.parquet`) to the `data/raw` directory
2. Process the data:
   - Filter and align cells between datasets
   - Normalize gene expression values
   - Extract gene names
3. Save processed data to the `data/processed` directory:
   - `source_data.npy`: Processed source dataset
   - `target_data.npy`: Processed target dataset
   - `source_genes.npy`: Gene names in source dataset
   - `target_genes.npy`: Gene names in target dataset

## Training

To train the model:

```bash
python src/main.py --train
```

Training process:
1. Loads processed data
2. Splits data into training and validation sets
3. Initializes the multi-encoder autoencoder model
4. Trains the model with:
   - AdamW optimizer
   - Learning rate scheduling
   - Gradient clipping
   - Early stopping
5. Saves model checkpoints to `models/checkpoints/`
6. Generates training metrics and visualizations in `results/`

Training parameters can be configured in `configs/default_config.json`.

## Evaluation

To evaluate a trained model:

```bash
python src/main.py --evaluate --model_path models/checkpoints/best_model_XXXXXXXX_XXXXXX.pt
```

Evaluation metrics:
- Mean Squared Error (MSE)
- R² score
- Pearson correlation (overall and per-gene)
- t-SNE visualization of latent space
- Gene-level performance analysis

## Project Structure

```
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── models/                   # Saved model checkpoints
│   └── checkpoints/          # Model checkpoints
├── notebooks/                # Jupyter notebooks for exploration
├── results/                  # Results and visualizations
│   ├── figures/              # Visualizations and plots
│   └── metrics/              # Performance metrics
├── src/                      # Source code
│   ├── data/                 # Data processing code
│   ├── models/               # Model definitions
│   ├── utils/                # Utility functions
│   ├── evaluate.py           # Evaluation script
│   ├── main.py               # Main script
│   └── train.py              # Training script
└── README.md                 # Project documentation
```

## Results

The model generates various visualizations and metrics:
- t-SNE visualizations of the latent space
- Distributions of gene-level correlations
- Encoder alignment analysis
- Per-gene performance metrics
- Sample-level prediction visualizations

## Future Work

Potential improvements to explore:
- Attention mechanisms to better capture gene-gene interactions
- Transfer learning from pre-trained gene expression models
- Integration of spatial information to improve predictions
- Exploration of different latent space dimensions and architectures