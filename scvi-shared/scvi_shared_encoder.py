import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import umap
from datetime import datetime
from tqdm import tqdm
import anndata
import scvi
import pandas as pd
import matplotlib.colors as mcolors

# Set precision for better GPU performance
torch.set_float32_matmul_precision('high')

class SCVISharedLatentEncoder(nn.Module):
    """
    A model that combines two pre-trained scVI encoders and projects them into a shared latent space.
    Uses MMD loss to align the distributions from both encoders.
    """
    def __init__(self, source_dim, target_dim, latent_dim=32, hidden_dim=64):
        super(SCVISharedLatentEncoder, self).__init__()
        
        # Define dimensions for the model
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Source encoder projection network
        # Takes scVI latent rep as input and projects to shared space
        self.source_projector = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Target encoder projection network
        # Takes scVI latent rep as input and projects to shared space
        self.target_projector = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim)
        )

    def project_source(self, x):
        return self.source_projector(x)
    
    def project_target(self, x):
        return self.target_projector(x)
    
    def forward(self, x_source, x_target=None):
        if x_target is not None:
            # Training mode with both inputs
            z_source = self.project_source(x_source)
            z_target = self.project_target(x_target)
            return z_source, z_target
        else:
            # Inference mode with only source input
            z_source = self.project_source(x_source)
            return z_source

def mmd_loss(x, y, kernel='rbf', sigma_list=[0.01, 0.1, 1, 10, 100]):
    """
    Maximum Mean Discrepancy (MMD) loss with multiple RBF kernels.
    This encourages the distributions to match across the entire space.
    
    Args:
        x: First sample
        y: Second sample
        kernel: Kernel type (only rbf supported)
        sigma_list: List of sigma values for the RBF kernels
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    # Expand the tensors to compute pairwise distances
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    
    # Compute pairwise squared Euclidean distances
    l2_distance_matrix = ((tiled_x - tiled_y) ** 2).sum(2)
    
    # Apply multiple RBF kernels and sum the results
    mmd_loss_value = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        
        # Compute kernel values
        xx_kernel = torch.exp(-gamma * ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(2))
        yy_kernel = torch.exp(-gamma * ((y.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2))
        xy_kernel = torch.exp(-gamma * l2_distance_matrix)
        
        # Compute means (we exclude diagonal elements for xx and yy kernels)
        xx_mean = (xx_kernel.sum() - torch.trace(xx_kernel)) / (x_size * (x_size - 1))
        yy_mean = (yy_kernel.sum() - torch.trace(yy_kernel)) / (y_size * (y_size - 1))
        xy_mean = xy_kernel.mean()
        
        # Add to total MMD loss
        mmd_loss_value += xx_mean + yy_mean - 2 * xy_mean
    
    return mmd_loss_value

def load_data(source_model_path, target_model_path, source_adata_path, target_adata_path, mapping_path):
    """
    Load data and prepare for training:
    1. Load the pre-trained scVI models
    2. Extract latent representations
    3. Prepare paired data based on mapping
    """
    print("Loading source and target scVI models...")
    
    # Load the pre-trained models (will download if needed)
    source_model = scvi.model.SCVI.load(source_model_path)
    target_model = scvi.model.SCVI.load(target_model_path)
    
    # Load anndata objects
    source_adata = sc.read_h5ad(source_adata_path)
    target_adata = sc.read_h5ad(target_adata_path)
    
    # Load cell mappings
    mapping_df = pd.read_csv(mapping_path)
    print(f"Loaded mapping file with {len(mapping_df)} rows")
    
    # Extract latent representations
    print("Extracting latent representations...")
    source_latent = source_model.get_latent_representation(source_adata)
    target_latent = target_model.get_latent_representation(target_adata)
    
    # Create dictionaries for cell name to index mapping
    source_cell_to_idx = {cell: idx for idx, cell in enumerate(source_adata.obs_names)}
    target_cell_to_idx = {cell: idx for idx, cell in enumerate(target_adata.obs_names)}
    
    # Create training pairs based on mapping
    X_source_paired = []
    X_target_paired = []
    source_cell_ids = []
    target_cell_ids = []
    
    for _, row in mapping_df.iterrows():
        source_id = row['cell_id_source']
        target_id = row['cell_id_target']
        
        if source_id in source_cell_to_idx and target_id in target_cell_to_idx:
            source_idx = source_cell_to_idx[source_id]
            target_idx = target_cell_to_idx[target_id]
            
            # Add this pair to our training data
            X_source_paired.append(source_latent[source_idx])
            X_target_paired.append(target_latent[target_idx])
            source_cell_ids.append(source_id)
            target_cell_ids.append(target_id)
    
    X_source_paired = np.stack(X_source_paired)
    X_target_paired = np.stack(X_target_paired)
    
    print(f"Final paired data shapes:")
    print(f"Source latent: {X_source_paired.shape}")
    print(f"Target latent: {X_target_paired.shape}")
    
    # Store cell metadata for later visualization
    cell_metadata = {
        'source_ids': source_cell_ids,
        'target_ids': target_cell_ids,
        'source_adata': source_adata,
        'target_adata': target_adata
    }
    
    return X_source_paired, X_target_paired, cell_metadata

def visualize_latent_space(shared_latent_source, shared_latent_target, cell_metadata, output_dir, title_suffix=""):
    """
    Visualize the shared latent space using UMAP.
    Color points by dataset source and by cell clusters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine both latent representations
    combined_latent = np.vstack([shared_latent_source, shared_latent_target])
    dataset_labels = np.array(['Source'] * len(shared_latent_source) + ['Target'] * len(shared_latent_target))
    
    # Get original cluster labels from AnnData objects, if they exist
    source_clusters = None
    target_clusters = None
    
    source_adata = cell_metadata['source_adata']
    target_adata = cell_metadata['target_adata']
    
    # Try to get cluster information from original anndata
    if 'leiden' in source_adata.obs:
        source_ids = cell_metadata['source_ids']
        source_id_to_idx = {id: i for i, id in enumerate(source_adata.obs_names)}
        source_indices = [source_id_to_idx.get(id) for id in source_ids if id in source_id_to_idx]
        source_clusters = source_adata.obs['leiden'].iloc[source_indices].values
    
    if 'leiden' in target_adata.obs:
        target_ids = cell_metadata['target_ids']
        target_id_to_idx = {id: i for i, id in enumerate(target_adata.obs_names)}
        target_indices = [target_id_to_idx.get(id) for id in target_ids if id in target_id_to_idx]
        target_clusters = target_adata.obs['leiden'].iloc[target_indices].values
    
    # Run UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    latent_umap = reducer.fit_transform(combined_latent)
    
    # Plot by dataset source (UMAP)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_umap[:, 0], latent_umap[:, 1], 
                         c=[0 if x == 'Source' else 1 for x in dataset_labels],
                         cmap='coolwarm', s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label='Dataset')
    plt.title(f'Shared Latent Space (UMAP) - By Dataset {title_suffix}')
    plt.savefig(f"{output_dir}/latent_umap_dataset{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    # Plot by cluster if we have that information
    if source_clusters is not None and target_clusters is not None:
        # Combine cluster labels from both datasets
        all_clusters = np.concatenate([source_clusters, target_clusters])
        
        # Create a color map for clusters
        unique_clusters = np.unique(all_clusters)
        n_clusters = len(unique_clusters)
        color_map = plt.cm.get_cmap('tab20', n_clusters)
        
        # Map cluster labels to numbers for coloring
        cluster_to_int = {cluster: i for i, cluster in enumerate(unique_clusters)}
        cluster_colors = [cluster_to_int[cluster] for cluster in all_clusters]
        
        # Plot UMAP by cluster
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(latent_umap[:, 0], latent_umap[:, 1], 
                            c=cluster_colors, cmap='tab20', s=5, alpha=0.7)
        plt.title(f'Shared Latent Space (UMAP) - By Cluster {title_suffix}')
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color_map(cluster_to_int[cluster]), 
                                    label=f'Cluster {cluster}') 
                         for cluster in unique_clusters]
        plt.legend(handles=legend_elements, title="Clusters", loc="upper right")
        plt.savefig(f"{output_dir}/latent_umap_clusters{title_suffix.replace(' ', '_')}.png", dpi=300)
        plt.close()
    
    # Split visualization to show source vs target alignment
    n_source = len(shared_latent_source)
    source_umap = latent_umap[:n_source]
    target_umap = latent_umap[n_source:]
    
    # Plot side-by-side UMAP
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    plt.scatter(source_umap[:, 0], source_umap[:, 1], c='blue', s=5, alpha=0.7, label='Source')
    plt.title('Source Dataset - UMAP')
    
    plt.subplot(1, 2, 2)
    plt.scatter(target_umap[:, 0], target_umap[:, 1], c='red', s=5, alpha=0.7, label='Target')
    plt.title('Target Dataset - UMAP')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latent_umap_comparison{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    # Create a joint visualization with transparency
    plt.figure(figsize=(12, 10))
    plt.scatter(source_umap[:, 0], source_umap[:, 1], c='blue', s=5, alpha=0.5, label='Source')
    plt.scatter(target_umap[:, 0], target_umap[:, 1], c='red', s=5, alpha=0.5, label='Target')
    plt.legend()
    plt.title(f'Shared Latent Space (UMAP) - Joint View {title_suffix}')
    plt.savefig(f"{output_dir}/latent_umap_joint{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    # Return the latent embeddings for further analysis
    return {
        'umap': latent_umap,
        'n_source': n_source
    }

def train_shared_latent_model(config):
    """
    Train the shared latent space model.
    """
    # Create output directories
    output_dir = config['output_dir']
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    # Generate a run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    X_source, X_target, cell_metadata = load_data(
        config['source_model_path'],
        config['target_model_path'],
        config['source_adata_path'],
        config['target_adata_path'],
        config['mapping_path']
    )
    
    # Create dataset and data loaders
    dataset = TensorDataset(
        torch.FloatTensor(X_source),
        torch.FloatTensor(X_target)
    )
    
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    # Initialize the model
    model = SCVISharedLatentEncoder(
        source_dim=X_source.shape[1],
        target_dim=X_target.shape[1],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim']
    )
    
    # Set up device (GPU/MPS if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model.to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping_counter = 0
    
    # Train the model
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}") as pbar:
            for batch_idx, (x_source, x_target) in enumerate(train_loader):
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                z_source, z_target = model(x_source, x_target)
                
                # Calculate MMD loss
                loss = mmd_loss(z_source, z_target)
                
                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x_source, x_target in val_loader:
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                
                z_source, z_target = model(x_source, x_target)
                val_loss += mmd_loss(z_source, z_target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_dir}/models/best_model_{run_id}.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
            early_stopping_counter = 0
            
            # Visualize the latent space for the best model
            if (epoch + 1) % config['visualize_every'] == 0:
                # Get all latent representations
                all_source_latent = []
                all_target_latent = []
                
                with torch.no_grad():
                    for x_source, x_target in DataLoader(dataset, batch_size=config['batch_size']):
                        x_source = x_source.to(device)
                        x_target = x_target.to(device)
                        
                        z_source, z_target = model(x_source, x_target)
                        all_source_latent.append(z_source.cpu().numpy())
                        all_target_latent.append(z_target.cpu().numpy())
                
                all_source_latent = np.vstack(all_source_latent)
                all_target_latent = np.vstack(all_target_latent)
                
                # Visualize the latent space
                visualize_latent_space(
                    all_source_latent, 
                    all_target_latent, 
                    cell_metadata,
                    f"{output_dir}/results",
                    f"Epoch {epoch+1}"
                )
        else:
            early_stopping_counter += 1
            
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Check for early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MMD Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/results/loss_curve_{run_id}.png", dpi=300)
    plt.close()
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(f"{output_dir}/models/best_model_{run_id}.pt"))
    model.eval()
    
    # Final latent space visualization with the best model
    all_source_latent = []
    all_target_latent = []
    
    with torch.no_grad():
        for x_source, x_target in DataLoader(dataset, batch_size=config['batch_size']):
            x_source = x_source.to(device)
            x_target = x_target.to(device)
            
            z_source, z_target = model(x_source, x_target)
            all_source_latent.append(z_source.cpu().numpy())
            all_target_latent.append(z_target.cpu().numpy())
    
    all_source_latent = np.vstack(all_source_latent)
    all_target_latent = np.vstack(all_target_latent)
    
    # Final visualization
    visualize_latent_space(
        all_source_latent, 
        all_target_latent, 
        cell_metadata,
        f"{output_dir}/results",
        "Final"
    )
    
    # Save embeddings for later use
    np.save(f"{output_dir}/results/source_latent_embeddings_{run_id}.npy", all_source_latent)
    np.save(f"{output_dir}/results/target_latent_embeddings_{run_id}.npy", all_target_latent)
    
    return model, run_id

if __name__ == "__main__":
    # Check device availability
    print("\nDevice information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Configuration
    config = {
        # Model paths
        'source_model_path': 'source_scvi_model',
        'target_model_path': 'target_scvi_model',
        'source_adata_path': 'data/sq_cell_feature_1.h5ad',
        'target_adata_path': 'data/sq_cell_feature_2.h5ad',
        'mapping_path': 'data/processed/cell_mapping.csv',
        
        # Model parameters
        'latent_dim': 32,       # Dimension of shared latent space
        'hidden_dim': 64,       # Hidden layer dimension
        
        # Training parameters
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'train_split': 0.8,
        'early_stopping_patience': 10,
        'grad_clip': 5.0,
        'visualize_every': 5,   # Visualize latent space every N epochs
        
        # Output directory
        'output_dir': 'scvi-shared'
    }
    
    # Train the model
    model, run_id = train_shared_latent_model(config)
    
    print(f"\nTraining complete! Run ID: {run_id}")
    print(f"Results saved in {config['output_dir']}/results")
    print(f"Best model saved in {config['output_dir']}/models/best_model_{run_id}.pt")