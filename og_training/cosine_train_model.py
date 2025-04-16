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

from models.multi_encoder import MultiEncoderAutoencoder

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

def train_model(config):
    """Train the multi-encoder autoencoder model."""
    # Create directories
    os.makedirs('newAfterAnnot/models', exist_ok=True)
    os.makedirs('newAfterAnnot/results', exist_ok=True)
    
    # Create run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and prepare data
    X_source, X_target, source_genes, target_genes = load_and_prepare_data(
        config['data_paths']['source_h5ad'],
        config['data_paths']['target_h5ad'],
        'data/processed/cell_mapping.csv'
    )
    
    # Create dataset
    dataset = TensorDataset(
        torch.FloatTensor(X_source),
        torch.FloatTensor(X_target)
    )
    
    # Split into train and validation
    train_size = int(config['training']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Initialize model
    model = MultiEncoderAutoencoder(
        source_dim=X_source.shape[1],
        target_dim=X_target.shape[1],
        latent_dim=config['model']['latent_dim'],
        hidden_dim=config['model']['hidden_dim']
    )
    
    # Setup training device - use MPS if available, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    reconstruction_criterion = nn.SmoothL1Loss(beta=0.1)
    latent_criterion = nn.CosineEmbeddingLoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    pearson_corrs = []
    
    print(f"\nStarting training on {device}")
    print(f"Source dimension: {X_source.shape[1]}")
    print(f"Target dimension: {X_target.shape[1]}")
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x_source, x_target) in enumerate(train_loader):
            x_source = x_source.to(device)
            x_target = x_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_from_source, recon_from_target, z_source, z_target = model(x_source, x_target)
            
            # Calculate losses
            recon_loss_source = reconstruction_criterion(recon_from_source, x_target)
            recon_loss_target = reconstruction_criterion(recon_from_target, x_target)
            latent_loss = latent_criterion(z_source, z_target, 
                                         torch.ones(z_source.size(0)).to(device))
            
            # Combined loss
            loss = (config['training']['loss_weights']['recon_source'] * recon_loss_source +
                   config['training']['loss_weights']['recon_target'] * recon_loss_target +
                   config['training']['loss_weights']['latent'] * latent_loss)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
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
        
        # Calculate correlation
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_targets = np.concatenate([t.flatten() for t in all_targets])
        pearson_corr, _ = pearsonr(all_preds, all_targets)
        pearson_corrs.append(pearson_corr)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Correlation = {pearson_corr:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'new/models/best_model_{run_id}.pt')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'new/models/checkpoint_epoch_{epoch+1}_{run_id}.pt')
    
    # Save final model
    torch.save(model.state_dict(), f'new/models/final_model_{run_id}.pt')
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(pearson_corrs)
    plt.title('Pearson Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    
    plt.tight_layout()
    plt.savefig(f'new/results/training_curves_{run_id}.png')
    plt.close()
    
    # Save training config and results
    results = {
        'run_id': run_id,
        'train_losses': convert_to_serializable(train_losses),
        'val_losses': convert_to_serializable(val_losses),
        'pearson_correlations': convert_to_serializable(pearson_corrs),
        'best_val_loss': float(best_val_loss),
        'config': convert_to_serializable(config)
    }
    
    with open(f'new/results/training_results_{run_id}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
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
            'latent_dim': 384,
            'hidden_dim': 128
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 5,
            'early_stopping_patience': 5,
            'grad_clip': 5.0,
            'train_split': 0.7,
            'loss_weights': {
                'recon_source': 0.3,
                'recon_target': 0.7,
                'latent': 0.2
            }
        }
    }
    
    model, run_id = train_model(config)
    print(f"\nTraining complete! Run ID: {run_id}")
    print("Model saved in newAfterAnnot/models/")
    print("Results saved in newAfterAnot/results/")