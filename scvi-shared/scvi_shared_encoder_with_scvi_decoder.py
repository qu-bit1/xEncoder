import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

class SCVISharedLatentEncoderWithScVIDecoder(nn.Module):
    """
    A model that combines two pre-trained scVI encoders and a pre-trained scVI decoder.
    Projects source and target latent representations into a shared latent space,
    then uses the target scVI model's decoder to reconstruct target features.
    Uses MMD loss to align the distributions from both encoders.
    """
    def __init__(self, source_dim, target_dim, latent_dim=32, hidden_dim=64, target_scvi_model=None):
        super(SCVISharedLatentEncoderWithScVIDecoder, self).__init__()
        
        # Define dimensions for the model
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.target_scvi_model = target_scvi_model
        
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
        
        # Projection from shared latent space to target scVI latent space
        # This maps our shared latent space back to the format expected by scVI decoder
        self.latent_to_target_scvi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, target_dim)
        )
    
    def project_source(self, x):
        return self.source_projector(x)
    
    def project_target(self, x):
        return self.target_projector(x)
    
    def map_to_target_latent(self, z):
        """Map from shared latent space to target scVI latent space"""
        return self.latent_to_target_scvi(z)
    
    def decode(self, z):
        """Decode latent representation to target feature space using scVI decoder."""
        # First map the shared latent space to target scVI latent space
        z_target_scvi = self.map_to_target_latent(z)
        
        # Use the target scVI model's decoder
        # We need to put in inference mode and ensure no gradients are computed for scVI internals
        with torch.no_grad():
            self.target_scvi_model.module.eval()  # Set to eval mode
            
            # Get batch size
            batch_size = z_target_scvi.size(0)
            
            # Generate samples from the decoder using our mapped latent values
            # This reconstructs the gene expression from the latent space
            decoder_input = {
                'z': z_target_scvi,  # Latent space representation
                'library': torch.ones_like(z_target_scvi[:, :1]),  # Library size (set to 1 for normalization)
                'batch_index': torch.zeros(batch_size, dtype=torch.long, device=z_target_scvi.device)  # Assuming single batch
            }
            
            # Get the mean of the negative binomial distribution (expected gene expression)
            generative_outputs = self.target_scvi_model.module.generative(**decoder_input)
            
            # The structure of the output depends on the version of scVI
            # For newer versions, px is a distribution object, not a dictionary
            if hasattr(generative_outputs['px'], 'rate'):
                # Access the rate attribute directly from the distribution object
                decoded_gene_expression = generative_outputs['px'].rate
            elif isinstance(generative_outputs['px'], dict) and 'rate' in generative_outputs['px']:
                # For older versions where px is a dictionary with rate key
                decoded_gene_expression = generative_outputs['px']['rate']
            else:
                # Fallback to getting the mean of the distribution
                decoded_gene_expression = generative_outputs['px'].mean
            
        return decoded_gene_expression
    
    def forward(self, x_source, x_target=None):
        if x_target is not None:
            # Training mode with both inputs
            z_source = self.project_source(x_source)
            z_target = self.project_target(x_target)
            
            # Add decoding - both encoders should produce representations
            # that decode to the target features (higher dimensional space)
            recon_source = self.decode(z_source)  # Source encoder -> shared space -> target reconstruction
            recon_target = self.decode(z_target)  # Target encoder -> shared space -> target reconstruction
            
            return z_source, z_target, recon_source, recon_target
        else:
            # Inference mode with only source input
            z_source = self.project_source(x_source)
            recon_source = self.decode(z_source)
            return z_source, recon_source

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
    
    # Handle case where batch size is 1
    if x_size == 1 or y_size == 1:
        # Fallback to simpler implementation or MSE
        return F.mse_loss(x.mean(0), y.mean(0))
    
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
        xx_mean = (xx_kernel.sum() - torch.trace(xx_kernel)) / max(1, (x_size * (x_size - 1)))
        yy_mean = (yy_kernel.sum() - torch.trace(yy_kernel)) / max(1, (y_size * (y_size - 1)))
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
    print("Loading datasets and models...")
    
    # Load AnnData objects first
    source_adata = sc.read_h5ad(source_adata_path)
    target_adata = sc.read_h5ad(target_adata_path)
    
    # Setup AnnData for scVI
    scvi.model.SCVI.setup_anndata(source_adata)
    scvi.model.SCVI.setup_anndata(target_adata)
    
    # Load the pre-trained models with their corresponding AnnData objects
    source_model = scvi.model.SCVI.load(source_model_path, adata=source_adata)
    target_model = scvi.model.SCVI.load(target_model_path, adata=target_adata)
    
    # Load cell mappings
    mapping_df = pd.read_csv(mapping_path)
    print(f"Loaded mapping file with {len(mapping_df)} rows")
    
    # Extract latent representations
    print("Extracting latent representations...")
    source_latent = source_model.get_latent_representation()
    target_latent = target_model.get_latent_representation()
    
    # Create dictionaries for cell name to index mapping
    source_cell_to_idx = {cell: idx for idx, cell in enumerate(source_adata.obs_names)}
    target_cell_to_idx = {cell: idx for idx, cell in enumerate(target_adata.obs_names)}
    
    # Create training pairs based on mapping, handling one-to-many relationships
    X_source_paired = []
    X_target_paired = []
    source_cell_ids = []
    target_cell_ids = []
    
    # Count successful mappings for logging
    successful_mappings = 0
    total_mappings = len(mapping_df)
    unique_source_cells = 0
    source_cells_used = set()
    
    # Group by source_id to handle one-to-many mapping correctly
    source_to_targets = mapping_df.groupby('cell_id_source')['cell_id_target'].apply(list).to_dict()
    
    for source_id, target_id_list in source_to_targets.items():
        if source_id in source_cell_to_idx:
            source_idx = source_cell_to_idx[source_id]
            valid_target_pairs = 0
            
            # For each source cell, add a pair with each of its mapped target cells
            for target_id in target_id_list:
                if target_id in target_cell_to_idx:
                    target_idx = target_cell_to_idx[target_id]
                    
                    # Add this pair to our training data
                    X_source_paired.append(source_latent[source_idx])
                    X_target_paired.append(target_latent[target_idx])
                    source_cell_ids.append(source_id)
                    target_cell_ids.append(target_id)
                    valid_target_pairs += 1
                    successful_mappings += 1
            
            # Count unique source cells that had at least one valid target mapping
            if valid_target_pairs > 0:
                source_cells_used.add(source_id)
                unique_source_cells += 1
    
    print(f"Successfully matched {successful_mappings} cell pairs out of {total_mappings} mappings ({successful_mappings/total_mappings:.2%})")
    print(f"Used {unique_source_cells} unique source cells with an average of {successful_mappings/max(1, unique_source_cells):.2f} target cells per source cell")
    
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
        'target_adata': target_adata,
        'source_model': source_model,
        'target_model': target_model  # Store the target model for decoding
    }
    
    return X_source_paired, X_target_paired, cell_metadata

def visualize_latent_space(shared_latent_source, shared_latent_target, cell_metadata, output_dir, title_suffix=""):
    """
    Visualize the shared latent space using UMAP with scanpy plotting functions.
    Color points by dataset source and by cell clusters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an AnnData object to hold the combined data
    combined_latent = np.vstack([shared_latent_source, shared_latent_target])
    dataset_labels = np.array(['Source'] * len(shared_latent_source) + ['Target'] * len(shared_latent_target))
    
    # Create metadata for plotting
    obs_df = pd.DataFrame({
        'dataset': dataset_labels,
        'cell_type': np.zeros(len(combined_latent), dtype=str)  # Will be filled in if clusters exist
    })
    
    # Add cell IDs if available
    if 'source_ids' in cell_metadata and 'target_ids' in cell_metadata:
        cell_ids = np.concatenate([cell_metadata['source_ids'], cell_metadata['target_ids']])
        obs_df.index = cell_ids
    else:
        obs_df.index = [f'cell_{i}' for i in range(len(combined_latent))]
    
    # Create AnnData object
    adata = sc.AnnData(X=combined_latent, obs=obs_df)
    
    # Get original cluster labels from AnnData objects, if they exist
    source_adata = cell_metadata['source_adata']
    target_adata = cell_metadata['target_adata']
    
    # Try to get cluster information and fill in obs
    have_clusters = False
    if 'leiden' in source_adata.obs.columns and 'leiden' in target_adata.obs.columns:
        try:
            # Match source clusters
            source_ids = cell_metadata['source_ids']
            source_id_to_idx = {id: i for i, id in enumerate(source_adata.obs_names)}
            source_indices = [source_id_to_idx.get(id) for id in source_ids if id in source_id_to_idx]
            
            # Match target clusters
            target_ids = cell_metadata['target_ids']
            target_id_to_idx = {id: i for i, id in enumerate(target_adata.obs_names)}
            target_indices = [target_id_to_idx.get(id) for id in target_ids if id in target_id_to_idx]
            
            if source_indices and target_indices:
                source_clusters = source_adata.obs['leiden'].iloc[source_indices].values
                target_clusters = target_adata.obs['leiden'].iloc[target_indices].values
                
                # Combine clusters
                all_clusters = np.concatenate([source_clusters, target_clusters])
                adata.obs['cluster'] = all_clusters
                have_clusters = True
                print(f"Successfully matched clusters for {len(source_indices)} source cells and {len(target_indices)} target cells")
        except Exception as e:
            print(f"Error matching clusters: {e}")
            have_clusters = False
    
    # Run UMAP with scanpy with parameters to handle potential spectral initialization issues
    print("Running UMAP...")
    # Add a small amount of random noise to prevent spectral initialization issues
    adata_copy = adata.copy()
    noise = np.random.normal(0, 0.0001, size=adata_copy.X.shape)
    adata_copy.X = adata_copy.X + noise
    
    # Use more robust parameters for neighbor computation
    sc.pp.neighbors(adata_copy, use_rep='X', n_neighbors=30, method='umap')
    
    # Set UMAP parameters to avoid spectral initialization issues
    sc.tl.umap(adata_copy, min_dist=0.3, spread=1.0, random_state=42, 
               init_pos='random', n_components=2)
    
    # Plot the UMAPs
    sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(8, 8))
    
    # Set the scanpy figures directory to our output directory
    sc.settings.figdir = output_dir
    
    # Plot by dataset
    print("Generating dataset UMAP plot...")
    sc.pl.umap(adata_copy, color='dataset', 
               title=f'Shared Latent Space - By Dataset {title_suffix}',
               palette={'Source': 'blue', 'Target': 'red'},
               size=30, alpha=0.7, legend_loc='on data',
               save=f"dataset{title_suffix.replace(' ', '_')}.png")
    
    # Plot by clusters if available
    if have_clusters:
        print("Generating cluster UMAP plot...")
        # Copy cluster info to the noise-added object
        adata_copy.obs['cluster'] = adata.obs['cluster']
        sc.pl.umap(adata_copy, color='cluster', 
                   title=f'Shared Latent Space - By Cluster {title_suffix}',
                   palette='tab20', size=30, alpha=0.7, legend_loc='on data',
                   save=f"clusters{title_suffix.replace(' ', '_')}.png")
    
    # Create separate AnnData objects for source and target using the noise-added data
    n_source = len(shared_latent_source)
    adata_source = adata_copy[:n_source].copy()
    adata_target = adata_copy[n_source:].copy()
    
    # Side-by-side plots using scanpy's function
    print("Generating side-by-side comparison...")
    
    # Source dataset
    sc.pl.umap(adata_source, title=f'Source Dataset {title_suffix}',
               size=30, alpha=0.7, color='dataset',
               save=f"source{title_suffix.replace(' ', '_')}.png")
    
    # Target dataset
    sc.pl.umap(adata_target, title=f'Target Dataset {title_suffix}',
               size=30, alpha=0.7, color='dataset',
               save=f"target{title_suffix.replace(' ', '_')}.png")
    
    # Create a joint visualization with scanpy
    print("Generating joint visualization...")
    with plt.rc_context({'figure.figsize': (12, 10)}):
        sc.pl.umap(adata_copy, color='dataset', palette={'Source': 'blue', 'Target': 'red'},
                   title=f'Shared Latent Space - Joint View {title_suffix}',
                   alpha=0.6, size=30, legend_loc='on data',
                   save=f"joint{title_suffix.replace(' ', '_')}.png")
    
    print(f"All UMAP visualizations saved to {output_dir}")
    
    # Return the latent embeddings for further analysis
    return {
        'umap': adata_copy.obsm['X_umap'],
        'n_source': n_source,
        'adata': adata_copy
    }

def compute_cluster_consistency_loss(z_source, z_target, source_labels, target_labels):
    """Compute loss to maintain cluster relationships."""
    # Convert labels to one-hot encodings
    unique_labels = torch.unique(torch.cat([source_labels, target_labels]))
    n_clusters = len(unique_labels)
    
    source_onehot = F.one_hot(source_labels, n_clusters).float()
    target_onehot = F.one_hot(target_labels, n_clusters).float()
    
    # Compute cluster centroids
    source_centroids = torch.matmul(source_onehot.t(), z_source) / (source_onehot.sum(0, keepdim=True).t() + 1e-10)
    target_centroids = torch.matmul(target_onehot.t(), z_target) / (target_onehot.sum(0, keepdim=True).t() + 1e-10)
    
    # Compute centroid alignment loss
    centroid_loss = F.mse_loss(source_centroids, target_centroids)
    
    return centroid_loss

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
    
    # Create dataset with cell IDs for cluster matching
    source_cell_ids_np = np.array(cell_metadata['source_ids'])
    target_cell_ids_np = np.array(cell_metadata['target_ids'])
    
    # Create dictionaries for cell name to index mapping 
    source_cell_to_idx = {cell: idx for idx, cell in enumerate(cell_metadata['source_adata'].obs_names)}
    target_cell_to_idx = {cell: idx for idx, cell in enumerate(cell_metadata['target_adata'].obs_names)}
    
    # Custom dataset for handling cell IDs
    class CellDataset(torch.utils.data.Dataset):
        def __init__(self, source_data, target_data, source_ids, target_ids):
            self.source_data = torch.FloatTensor(source_data)
            self.target_data = torch.FloatTensor(target_data)
            self.source_ids = source_ids
            self.target_ids = target_ids
            
        def __len__(self):
            return len(self.source_data)
        
        def __getitem__(self, idx):
            return (self.source_data[idx], self.target_data[idx], 
                    self.source_ids[idx], self.target_ids[idx])
    
    dataset = CellDataset(X_source, X_target, source_cell_ids_np, target_cell_ids_np)
    
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
    
    # Access the target scVI model's module
    target_scvi_model = cell_metadata['target_model'].module
    
    # Initialize the model with target scVI decoder
    model = SCVISharedLatentEncoderWithScVIDecoder(
        source_dim=X_source.shape[1],
        target_dim=X_target.shape[1],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        target_scvi_model=cell_metadata['target_model']
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
            for batch_idx, batch_data in enumerate(train_loader):
                x_source, x_target, batch_source_ids, batch_target_ids = batch_data
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with reconstruction
                z_source, z_target, recon_source, recon_target = model(x_source, x_target)
                
                # Calculate losses
                # MMD loss to align the latent distributions from both encoders
                mmd_loss_val = mmd_loss(z_source, z_target)
                
                # Reconstruction loss using the latent representation alignment
                # Since we're using scVI's decoder, the reconstructions are in gene expression space (5001 dims)
                # But our x_target is still in scVI latent space (10 dims)
                # So we compute the loss on the shared latent space instead
                
                # Map z_target back to target scVI latent space for comparison with x_target
                z_target_projection = model.map_to_target_latent(z_target)
                recon_loss = F.mse_loss(z_target_projection, x_target)
                
                # Get cluster labels if available
                cluster_loss = torch.tensor(0.0).to(device)
                if 'leiden' in cell_metadata['source_adata'].obs.columns and \
                   'leiden' in cell_metadata['target_adata'].obs.columns:
                    source_indices = [source_cell_to_idx[id] for id in batch_source_ids if id in source_cell_to_idx]
                    target_indices = [target_cell_to_idx[id] for id in batch_target_ids if id in target_cell_to_idx]
                    
                    if source_indices and target_indices:
                        source_labels = torch.tensor([cell_metadata['source_adata'].obs['leiden'].cat.codes.iloc[idx] 
                                                     for idx in source_indices], dtype=torch.long).to(device)
                        target_labels = torch.tensor([cell_metadata['target_adata'].obs['leiden'].cat.codes.iloc[idx] 
                                                     for idx in target_indices], dtype=torch.long).to(device)
                        cluster_loss = compute_cluster_consistency_loss(z_source, z_target, source_labels, target_labels)
                
                # Combined loss
                loss = (config['loss_weights']['mmd'] * mmd_loss_val + 
                       config['loss_weights']['reconstruction'] * recon_loss +
                       config['loss_weights']['cluster'] * cluster_loss)
                
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
            for batch_data in val_loader:
                x_source, x_target, _, _ = batch_data
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                
                z_source, z_target, recon_source, recon_target = model(x_source, x_target)
                val_loss += mmd_loss(z_source, z_target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_dir}/models/best_model_scvi_decoder_{run_id}.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
            early_stopping_counter = 0
            
            # Visualize the latent space for the best model
            if (epoch + 1) % config['visualize_every'] == 0:
                # Get all latent representations
                all_source_latent = []
                all_target_latent = []
                
                with torch.no_grad():
                    for batch_data in DataLoader(dataset, batch_size=config['batch_size']):
                        x_source, x_target, _, _ = batch_data
                        x_source = x_source.to(device)
                        x_target = x_target.to(device)
                        
                        z_source, z_target, recon_source, recon_target = model(x_source, x_target)
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
                    f"Epoch {epoch+1} (scVI Decoder)"
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
    plt.title('Training and Validation Loss (scVI Decoder)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/results/loss_curve_scvi_decoder_{run_id}.png", dpi=300)
    plt.close()
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(f"{output_dir}/models/best_model_scvi_decoder_{run_id}.pt"))
    model.eval()
    
    # Final latent space visualization with the best model
    all_source_latent = []
    all_target_latent = []
    
    with torch.no_grad():
        for batch_data in DataLoader(dataset, batch_size=config['batch_size']):
            x_source, x_target, _, _ = batch_data
            x_source = x_source.to(device)
            x_target = x_target.to(device)
            
            z_source, z_target, recon_source, recon_target = model(x_source, x_target)
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
        "Final (scVI Decoder)"
    )
    
    # Save embeddings for later use
    np.save(f"{output_dir}/results/source_latent_embeddings_scvi_decoder_{run_id}.npy", all_source_latent)
    np.save(f"{output_dir}/results/target_latent_embeddings_scvi_decoder_{run_id}.npy", all_target_latent)
    
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
        'mapping_path': 'data/cell_mapping.csv',
        
        # Model parameters
        'latent_dim': 32,       # Shared latent dimension
        'hidden_dim': 128,      # Hidden layer dimension for projection networks
        
        # Training parameters
        'batch_size': 256,      # Batch size for training
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'train_split': 0.8,
        'early_stopping_patience': 15,  # Increased patience for better convergence
        'grad_clip': 5.0,
        'visualize_every': 5,   # Visualize latent space every N epochs
        
        # Loss weights
        'loss_weights': {
            'mmd': 1.0,          # Maximum Mean Discrepancy loss weight
            'reconstruction': 1.0, # Increased reconstruction weight
            'cluster': 0.2       # Increased cluster consistency weight
        },
        
        # Output directory
        'output_dir': 'scvi-shared'
    }
    
    # Train the model
    model, run_id = train_shared_latent_model(config)
    
    print(f"\nTraining complete! Run ID: {run_id}")
    print(f"Results saved in {config['output_dir']}/results")
    print(f"Best model saved in {config['output_dir']}/models/best_model_scvi_decoder_{run_id}.pt")