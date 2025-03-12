import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import project modules
from models.multi_encoder import MultiEncoderAutoencoder
from utils.analysis import (
    calculate_gene_correlations,
    analyze_encoder_alignment,
    visualize_latent_space,
    noise_sensitivity_test
)

def compute_latent_loss(z_source, z_target, criterion):
    """Compute loss between source and target latent representations"""
    ones = torch.ones(z_source.size(0)).to(z_source.device)  # Cosine loss expects target +1 for similarity
    return criterion(z_source, z_target, ones)

def train_model(config):
    """
    Train the multi-encoder autoencoder model.
    
    Args:
        config: Dictionary containing training configuration
    """
    # Create directories for saving model, results, and analysis
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Create a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'runs/experiment_{run_id}')
    
    # Load data
    X_source = np.load(config['data_paths']['source_data'])
    X_target = np.load(config['data_paths']['target_data'])
    
    try:
        source_genes = np.load(config['data_paths']['source_genes'])
        target_genes = np.load(config['data_paths']['target_genes'])
    except:
        print("Gene name files not found. Proceeding without gene names.")
        source_genes = np.array([f"gene_{i}" for i in range(X_source.shape[1])])
        target_genes = np.array([f"gene_{i}" for i in range(X_target.shape[1])])
    
    print(f"Source data shape: {X_source.shape}")
    print(f"Target data shape: {X_target.shape}")
    
    # Common genes analysis
    common_genes = np.intersect1d(source_genes, target_genes)
    if len(common_genes) > 0:
        print(f"Number of common genes: {len(common_genes)}")
        
        # Get indices of common genes in both datasets
        source_indices = np.array([np.where(source_genes == gene)[0][0] for gene in common_genes])
        target_indices = np.array([np.where(target_genes == gene)[0][0] for gene in common_genes])
        
        # Calculate correlation for common genes
        corrs = []
        for i in range(len(source_indices)):
            corr, _ = pearsonr(X_source[:, source_indices[i]], X_target[:, target_indices[i]])
            corrs.append(corr)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(corrs, kde=True)
        plt.title('Correlation Distribution for Common Genes')
        plt.xlabel('Pearson Correlation')
        plt.savefig('results/figures/common_genes_correlation.png')
    
    # Split data into training and validation sets
    dataset = TensorDataset(torch.FloatTensor(X_source), torch.FloatTensor(X_target))

    train_size = int(config['training']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model dimensions
    source_dim = X_source.shape[1]
    target_dim = X_target.shape[1]
    latent_dim = config['model']['latent_dim']
    hidden_dim = config['model']['hidden_dim']
    
    print(f"Source dimension: {source_dim}")
    print(f"Target dimension: {target_dim}")
    print(f"Latent dimension: {latent_dim}")
    
    # Loss functions
    reconstruction_criterion = nn.SmoothL1Loss(beta=0.1)  # Less sensitive to outliers
    latent_criterion = nn.CosineEmbeddingLoss()  # Ensures similar but not identical embeddings

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, optimizer, and scheduler
    model = MultiEncoderAutoencoder(
        source_dim=source_dim, 
        target_dim=target_dim, 
        latent_dim=latent_dim, 
        hidden_dim=hidden_dim
    )
    
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
    
    # Initialize arrays to store metrics
    train_losses = []
    val_losses = []
    pearson_corrs = []
    best_val_loss = float('inf')
    num_epochs = config['training']['num_epochs']
    no_improve_epochs = 0  # Tracks epochs without improvement
    early_stopping_patience = config['training']['early_stopping_patience']
    grad_clip = config['training']['grad_clip']  # Gradient clipping for stability

    # Create a dictionary to store the experiment configuration
    experiment_config = {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": config['training']['learning_rate'],
        "weight_decay": config['training']['weight_decay'],
        "scheduler": "ReduceLROnPlateau",
        "loss_functions": {
            "reconstruction": "SmoothL1Loss",
            "latent": "CosineEmbeddingLoss"
        },
        "loss_weights": {
            "recon_source": config['training']['loss_weights']['recon_source'],
            "recon_target": config['training']['loss_weights']['recon_target'],
            "latent": config['training']['loss_weights']['latent']
        },
        "training_samples": train_size,
        "validation_samples": val_size,
        "run_id": run_id
    }

    # Save the configuration
    with open(f'results/metrics/experiment_config_{run_id}.json', 'w') as f:
        json.dump(experiment_config, f, indent=4)

    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        batch_losses = []
        
        for batch_idx, (x_source, x_target) in enumerate(train_loader):
            x_source, x_target = x_source.to(device), x_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_from_source, recon_from_target, z_source, z_target = model(x_source, x_target)
            
            # Calculate losses
            recon_loss_source = reconstruction_criterion(recon_from_source, x_target)
            recon_loss_target = reconstruction_criterion(recon_from_target, x_target)
            latent_loss = compute_latent_loss(z_source, z_target, latent_criterion)
            
            # Combined loss (target reconstruction is more important)
            loss = (
                config['training']['loss_weights']['recon_source'] * recon_loss_source + 
                config['training']['loss_weights']['recon_target'] * recon_loss_target + 
                config['training']['loss_weights']['latent'] * latent_loss
            )

            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            batch_losses.append({
                'loss': loss.item(),
                'recon_source': recon_loss_source.item(),
                'recon_target': recon_loss_target.item(),
                'latent': latent_loss.item()
            })
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Batch/ReconSource', recon_loss_source.item(), global_step)
            writer.add_scalar('Batch/ReconTarget', recon_loss_target.item(), global_step)
            writer.add_scalar('Batch/LatentLoss', latent_loss.item(), global_step)
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Recon Source: {recon_loss_source.item():.4f}, "
                      f"Recon Target: {recon_loss_target.item():.4f}, Latent Loss: {latent_loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        all_targets, all_predictions = [], []
        all_source_latents, all_target_latents = [], []

        with torch.no_grad():
            for x_source, x_target in val_loader:
                x_source, x_target = x_source.to(device), x_target.to(device)
                
                # Get both encodings for analysis
                z_source = model.encode_source(x_source)
                z_target = model.encode_target(x_target)
                recon_from_source = model.decode(z_source)
                
                # Store latent vectors for visualization
                all_source_latents.append(z_source.cpu().numpy())
                all_target_latents.append(z_target.cpu().numpy())
                
                # Calculate validation loss
                val_loss = reconstruction_criterion(recon_from_source, x_target)
                epoch_val_loss += val_loss.item()
                
                # Store for Pearson correlation
                all_targets.append(x_target.cpu().numpy())
                all_predictions.append(recon_from_source.cpu().numpy())
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Compute Pearson correlation
        all_targets_flat = np.concatenate([arr.flatten() for arr in all_targets])
        all_predictions_flat = np.concatenate([arr.flatten() for arr in all_predictions])
        pearson_corr, _ = pearsonr(all_targets_flat, all_predictions_flat)
        pearson_corrs.append(pearson_corr)
        
        # Log to TensorBoard
        writer.add_scalar('Epoch/TrainLoss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/ValLoss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/PearsonCorr', pearson_corr, epoch)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Pearson Correlation: {pearson_corr:.4f}")
        
        # Every 10 epochs, perform additional analysis
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            # Combine latent vectors for visualization
            source_latents = np.vstack(all_source_latents)
            
            # Visualize with t-SNE
            tsne_result = visualize_latent_space(
                source_latents, 
                title=f"Latent Space t-SNE Epoch {epoch+1}"
            )
            
            # Latent feature distributions
            plt.figure(figsize=(15, 10))
            for i in range(min(16, source_latents.shape[1])):
                plt.subplot(4, 4, i+1)
                sns.histplot(source_latents[:, i], kde=True)
                plt.title(f"Dim {i+1}")
            plt.tight_layout()
            plt.savefig(f"results/figures/latent_dimensions_epoch_{epoch+1}.png", dpi=300)
            plt.close()
            
            # Analyze encoder alignment
            if len(all_target_latents) > 0:
                alignment_score, _, _ = analyze_encoder_alignment(model, val_loader, device)
                writer.add_scalar('Analysis/EncoderAlignment', alignment_score, epoch)
                
            # Calculate per-gene correlations
            gene_corrs, _, _ = calculate_gene_correlations(model, val_loader, device, target_dim)
            
            # Plot distribution of correlations
            plt.figure(figsize=(10, 6))
            sns.histplot(gene_corrs, kde=True)
            plt.xlabel('Pearson Correlation')
            plt.ylabel('Count')
            plt.title(f'Distribution of Gene-Level Correlations - Epoch {epoch+1}')
            plt.savefig(f"results/figures/gene_correlation_distribution_epoch_{epoch+1}.png", dpi=300)
            plt.close()
            
            # Find best and worst predicted genes
            top_genes_idx = np.argsort(gene_corrs)[-20:]
            bottom_genes_idx = np.argsort(gene_corrs)[:20]
            
            with open(f"results/metrics/gene_performance_epoch_{epoch+1}.txt", "w") as f:
                f.write("Top 20 Best Predicted Genes:\n")
                for idx in reversed(top_genes_idx):
                    f.write(f"{target_genes[idx]}: {gene_corrs[idx]:.4f}\n")
                
                f.write("\nBottom 20 Worst Predicted Genes:\n")
                for idx in bottom_genes_idx:
                    f.write(f"{target_genes[idx]}: {gene_corrs[idx]:.4f}\n")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/checkpoints/best_model_{run_id}.pt')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
            no_improve_epochs = 0  # Reset counter
        else:
            no_improve_epochs += 1  # Increase counter

        # Early stopping check
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {early_stopping_patience} epochs.")
            break  # Stop training
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'pearson': pearson_corr
            }, f'models/checkpoints/model_epoch_{epoch+1}_{run_id}.pt')

    # Save final model
    torch.save(model.state_dict(), f'models/checkpoints/final_model_{run_id}.pt')
    print("Training complete!")
    
    # Plot training and validation loss
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(pearson_corrs, label='Pearson Correlation')
    plt.title('Pearson Correlation Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/figures/training_metrics_{run_id}.png', dpi=300)
    
    return model, run_id

if __name__ == "__main__":
    # Default configuration
    config = {
        'data_paths': {
            'source_data': 'data/processed/source_data.npy',
            'target_data': 'data/processed/target_data.npy',
            'source_genes': 'data/processed/source_genes.npy',
            'target_genes': 'data/processed/target_genes.npy'
        },
        'model': {
            'latent_dim': 384,
            'hidden_dim': 128
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 50,
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
    
    # Train the model
    model, run_id = train_model(config)