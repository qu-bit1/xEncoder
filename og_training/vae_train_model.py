import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import scipy.sparse as sp
from datetime import datetime
import json
import scanpy as sc
from tqdm import tqdm

# from models.multi_encoder import MultiEncoderAutoencoder
from models.vae import MultiEncoderVAE

def load_and_prepare_data(source_path, target_path, mapping_path):
    """Load and prepare data with one-to-many cell mapping."""
    print("Loading datasets...")
    
    # Load mapping information
    mapping_df = pd.read_csv(mapping_path)
    print(f"Loaded mapping file with {len(mapping_df)} rows")
    
    # Load AnnData objects
    adata_source = sc.read_h5ad(source_path)
    adata_target = sc.read_h5ad(target_path)
    print(f"Source dataset shape: {adata_source.shape}")
    print(f"Target dataset shape: {adata_target.shape}")
    
    # Filter cells based on mapping
    source_cells = mapping_df['cell_id_source'].unique()
    target_cells = mapping_df['cell_id_target'].unique()
    print(f"Unique source cells in mapping: {len(source_cells)}")
    print(f"Unique target cells in mapping: {len(target_cells)}")
    
    adata_source = adata_source[adata_source.obs_names.isin(source_cells)].copy()
    adata_target = adata_target[adata_target.obs_names.isin(target_cells)].copy()
    
    # Convert to dense arrays
    X_source = adata_source.X.toarray() if sp.issparse(adata_source.X) else adata_source.X
    X_target = adata_target.X.toarray() if sp.issparse(adata_target.X) else adata_target.X
    
    # Create dictionaries for cell name to index mapping
    source_cell_to_idx = {cell: idx for idx, cell in enumerate(adata_source.obs_names)}
    target_cell_to_idx = {cell: idx for idx, cell in enumerate(adata_target.obs_names)}
    
    # Analyze the mapping relationships
    source_counts = mapping_df['cell_id_source'].value_counts()
    target_counts = mapping_df['cell_id_target'].value_counts()
    print(f"\nMapping statistics:")
    print(f"Source cells with multiple targets: {len(source_counts[source_counts > 1])}")
    print(f"Target cells with multiple sources: {len(target_counts[target_counts > 1])}")
    print(f"Max targets per source: {source_counts.max()}")
    print(f"Max sources per target: {target_counts.max()}")
    
    # Create training pairs based on mapping
    # For each source-target pair in the mapping, create a training example
    X_source_paired = []
    X_target_paired = []
    
    for _, row in mapping_df.iterrows():
        source_id = row['cell_id_source']
        target_id = row['cell_id_target']
        
        if source_id in source_cell_to_idx and target_id in target_cell_to_idx:
            source_idx = source_cell_to_idx[source_id]
            target_idx = target_cell_to_idx[target_id]
            
            # Add this pair to our training data
            X_source_paired.append(X_source[source_idx])
            X_target_paired.append(X_target[target_idx])
    
    X_source_paired = np.stack(X_source_paired)
    X_target_paired = np.stack(X_target_paired)
    
    print(f"\nFinal paired data shapes:")
    print(f"Source: {X_source_paired.shape}")
    print(f"Target: {X_target_paired.shape}")
    print(f"(Each row represents one source-target cell pair from the mapping)")
    
    return (X_source_paired, X_target_paired, 
            adata_source.var_names.tolist(), adata_target.var_names.tolist())

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.smooth_l1_loss(recon_x, x, reduction="mean")  # Smooth L1 loss for robustness
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # KL divergence
    return recon_loss + 0.1 * kl_loss  # Weighted KL loss for better clustering

def visualize_latent_space(latent_vectors, save_path):
    import umap
    reducer = umap.UMAP(n_components=2)
    reduced = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], alpha=0.5, s=10)
    plt.title("Latent Space Visualization (UMAP)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(config):
    os.makedirs('VAE_L512/models', exist_ok=True)
    os.makedirs('VAE_L512/results', exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    X_source, X_target, source_genes, target_genes = load_and_prepare_data(
        config['data_paths']['source_h5ad'],
        config['data_paths']['target_h5ad'],
        'data/processed/cell_mapping.csv'
    )

    dataset = TensorDataset(
        torch.FloatTensor(X_source),
        torch.FloatTensor(X_target)
    )

    train_size = int(config['training']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model = MultiEncoderVAE(
        source_dim=X_source.shape[1],
        target_dim=X_target.shape[1],
        latent_dim=config['model']['latent_dim'],
        hidden_dim=config['model']['hidden_dim']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    all_latents = []
    patience = config['training'].get('early_stopping_patience', 10)
    patience_counter = 0

    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}") as pbar:
            for batch_idx, (x_source, x_target) in enumerate(train_loader):
                x_source = x_source.to(device)
                x_target = x_target.to(device)

                optimizer.zero_grad()

                recon_x, z, mu, logvar = model(x_source)
                loss = vae_loss(recon_x, x_target, mu, logvar)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                if epoch == config['training']['num_epochs'] - 1:
                    all_latents.append(z.detach().cpu())

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x_source, x_target in val_loader:
                x_source = x_source.to(device)
                x_target = x_target.to(device)

                recon_x, z, mu, logvar = model(x_source)
                val_loss += vae_loss(recon_x, x_target, mu, logvar).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'VAE_L512/models/best_model_{run_id}.pt')
            print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(f'VAE_L512/results/loss_curve_{run_id}.png')
    plt.close()

    if all_latents:
        latent_matrix = torch.cat(all_latents, dim=0).numpy()
        visualize_latent_space(latent_matrix, f'VAE_L512/results/latent_space_{run_id}.png')

    return model, run_id

if __name__ == "__main__":
    print("\nDevice information:")
    print(f"cuda available : {torch.cuda.is_available()} ")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS backend built: {torch.backends.mps.is_built()}")

    config = {
        'data_paths': {
            'source_h5ad': 'sq_cell_feature_1.h5ad',
            'target_h5ad': 'sq_cell_feature_2.h5ad'
        },
        'model': {
            'latent_dim': 512,
            'hidden_dim': 256
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'grad_clip': 5.0,
            'train_split': 0.7
        }
    }

    model, run_id = train_model(config)
    print(f"\nTraining complete! Run ID: {run_id}")
    print("Model saved in VAE_L512/models/")
    print("Results saved in VAE_L512/results/")
