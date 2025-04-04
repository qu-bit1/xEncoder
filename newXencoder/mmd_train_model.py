import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
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
from models.deep_multi_encoder import MultiEncoderAutoencoder

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

def mmd_loss(x, y, kernel='rbf', sigma=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) loss between two distributions."""
    def compute_kernel(x, y, sigma):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
        tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
        kernel_val = torch.exp(-((tiled_x - tiled_y) ** 2).sum(2) / (2 * sigma ** 2))
        return kernel_val
    
    xx = compute_kernel(x, x, sigma).mean()
    yy = compute_kernel(y, y, sigma).mean()
    xy = compute_kernel(x, y, sigma).mean()
    return xx + yy - 2 * xy

def train_model(config):
    os.makedirs('deepMMD_L_512/models', exist_ok=True)
    os.makedirs('deepMMD_L_512/results', exist_ok=True)
    
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
    
    model = MultiEncoderAutoencoder(
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    reconstruction_criterion = nn.SmoothL1Loss(beta=0.1)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    pearson_corrs = []
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}") as pbar:
            for batch_idx, (x_source, x_target) in enumerate(train_loader):
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                
                optimizer.zero_grad()
                
                recon_from_source, recon_from_target, z_source, z_target = model(x_source, x_target)
                
                recon_loss_source = reconstruction_criterion(recon_from_source, x_target)
                recon_loss_target = reconstruction_criterion(recon_from_target, x_target)
                mmd_loss_value = mmd_loss(z_source, z_target, sigma=1.0)
                
                loss = (config['training']['loss_weights']['recon_source'] * recon_loss_source +
                       config['training']['loss_weights']['recon_target'] * recon_loss_target +
                       config['training']['loss_weights']['mmd'] * mmd_loss_value)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_source, x_target in val_loader:
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                
                recon_from_source = model(x_source)[0]
                val_loss += reconstruction_criterion(recon_from_source, x_target).item()
                
                all_preds.append(recon_from_source.cpu().numpy())
                all_targets.append(x_target.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'deepMMD_L_512/models/best_model_{run_id}.pt')
            print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break
        
        scheduler.step(avg_val_loss)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(f'deepMMD_L_512/results/loss_curve_{run_id}.png')
    plt.close()
    
    return model, run_id



if __name__ == "__main__":
    # Check MPS availability
    print("\nDevice information:")
    print(f"cuda available : {torch.cuda.is_available()} ")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS backend built: {torch.backends.mps.is_built()}")
    
    # Configuration
    config = {
        'data_paths': {
            'source_h5ad': 'sq_cell_feature_1.h5ad',
            'target_h5ad': 'sq_cell_feature_2.h5ad'
        },
        'model': {
            'latent_dim': 20,
            'hidden_dim': 2
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'grad_clip': 5.0,
            'train_split': 0.7,
            'loss_weights': {
                'recon_source': 0.3,
                'recon_target': 0.7,
                'mmd': 0.1  
            }
        }
    }
    
    model, run_id = train_model(config)
    print(f"\nTraining complete! Run ID: {run_id}")
    print("Model saved in deepMMD_L_512/models/")
    print("Results saved in deepMMD_L_512/results/")