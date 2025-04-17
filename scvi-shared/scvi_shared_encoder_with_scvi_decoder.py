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
            
            # Access the rate attribute more robustly
            if hasattr(generative_outputs['px'], 'rate'):
                # For newer scVI versions that use distribution objects
                decoded_gene_expression = generative_outputs['px'].rate
            elif isinstance(generative_outputs['px'], dict) and 'rate' in generative_outputs['px']:
                # For older versions that use dictionaries
                decoded_gene_expression = generative_outputs['px']['rate']
            else:
                # Fallback to getting the mean of the distribution
                decoded_gene_expression = generative_outputs['px'].mean
            
        return decoded_gene_expression
    
    def forward(self, x_source, x_target=None):
        # Training mode with both inputs
        if x_target is not None:
            z_source = self.project_source(x_source)
            z_target = self.project_target(x_target)
            
            # Add decoding - both encoders should produce representations
            # that decode to the target features (higher dimensional space)
            recon_source = self.decode(z_source)  # Source encoder -> shared space -> target reconstruction
            recon_target = self.decode(z_target)  # Target encoder -> shared space -> target reconstruction
            
            return z_source, z_target, recon_source, recon_target
        # Inference mode with only source input
        else:
            z_source = self.project_source(x_source)
            recon_source = self.decode(z_source)
            return z_source, recon_source


def mmd_loss(x, y, kernel='rbf', sigma_list=None):
    """
    Maximum Mean Discrepancy (MMD) loss with multiple RBF kernels.
    This encourages the distributions to match across the entire space.
    
    Args:
        x: First sample
        y: Second sample
        kernel: Kernel type (only rbf supported)
        sigma_list: List of sigma values for the RBF kernels
    """
    if sigma_list is None:
        # Default sigma values spanning multiple scales
        sigma_list = [0.01, 0.1, 1, 10, 100]
        
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    # Handle case where batch size is 1
    if x_size == 1 or y_size == 1:
        # Fallback to simpler implementation or MSE
        return F.mse_loss(x.mean(0), y.mean(0))
    
    # Compute pairwise distances more efficiently
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    # Compute the squared norms
    x_norm = torch.diag(xx)
    y_norm = torch.diag(yy)
    
    # Compute pairwise squared distances using kernel trick
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
    dist_xx = x_norm.unsqueeze(1) + x_norm.unsqueeze(0) - 2 * xx
    dist_yy = y_norm.unsqueeze(1) + y_norm.unsqueeze(0) - 2 * yy
    dist_xy = x_norm.unsqueeze(1) + y_norm.unsqueeze(0) - 2 * xy
    
    # Apply multiple RBF kernels and sum the results
    mmd_loss_value = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        
        # Compute kernel values
        k_xx = torch.exp(-gamma * dist_xx)
        k_yy = torch.exp(-gamma * dist_yy)
        k_xy = torch.exp(-gamma * dist_xy)
        
        # Compute means (excluding diagonal elements for xx and yy kernels)
        k_xx_sum = (k_xx.sum() - torch.trace(k_xx)) / (x_size * (x_size - 1))
        k_yy_sum = (k_yy.sum() - torch.trace(k_yy)) / (y_size * (y_size - 1))
        k_xy_sum = k_xy.mean()
        
        # Add to total MMD loss
        mmd_loss_value += k_xx_sum + k_yy_sum - 2 * k_xy_sum
    
    return mmd_loss_value


def load_data(source_model_path, target_model_path, decoder_model_path, source_adata_path, target_adata_path, mapping_path):
    """
    Load data and prepare for training:
    1. Load the pre-trained scVI models
    2. Extract latent representations and gene expression
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
    decoder_model = scvi.model.SCVI.load(decoder_model_path, adata=target_adata)
    
    # Load cell mappings
    mapping_df = pd.read_csv(mapping_path)
    print(f"Loaded mapping file with {len(mapping_df)} rows")
    
    # Extract latent representations and gene expression
    print("Extracting latent representations and gene expression...")
    source_latent = source_model.get_latent_representation()
    target_latent = target_model.get_latent_representation()
    
    # Handle sparse matrix if needed
    if isinstance(target_adata.X, np.ndarray):
        target_expression = target_adata.X
    else:
        # Convert sparse to dense array
        target_expression = target_adata.X.toarray()
    
    # Create dictionaries for cell name to index mapping
    source_cell_to_idx = {cell: idx for idx, cell in enumerate(source_adata.obs_names)}
    target_cell_to_idx = {cell: idx for idx, cell in enumerate(target_adata.obs_names)}
    
    # Create training pairs based on mapping, handling one-to-many relationships
    X_source_paired = []
    X_target_paired = []
    Y_target_paired = []  # Store target gene expression
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
                    Y_target_paired.append(target_expression[target_idx])  # Add target gene expression
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
    
    # Convert to numpy arrays
    X_source_paired = np.stack(X_source_paired)
    X_target_paired = np.stack(X_target_paired)
    Y_target_paired = np.stack(Y_target_paired)
    
    print(f"Final paired data shapes:")
    print(f"Source latent: {X_source_paired.shape}")
    print(f"Target latent: {X_target_paired.shape}")
    print(f"Target expression: {Y_target_paired.shape}")
    
    # Store cell metadata for later visualization
    cell_metadata = {
        'source_ids': source_cell_ids,
        'target_ids': target_cell_ids,
        'source_adata': source_adata,
        'target_adata': target_adata
    }
    
    return X_source_paired, X_target_paired, Y_target_paired, cell_metadata, decoder_model


def visualize_latent_space(shared_latent_source, shared_latent_target, cell_metadata, output_dir, title_suffix=""):
    """
    Visualize the shared latent space using UMAP with scanpy plotting functions.
    Color points by cell clusters for source, target, and joint datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an AnnData object to hold the combined data
    combined_latent = np.vstack([shared_latent_source, shared_latent_target])
    dataset_labels = np.array(['Source'] * len(shared_latent_source) + ['Target'] * len(shared_latent_target))
    
    # Create metadata for plotting
    obs_df = pd.DataFrame({
        'dataset': dataset_labels
    })
    
    # Add cell IDs if available
    if 'source_ids' in cell_metadata and 'target_ids' in cell_metadata:
        cell_ids = np.concatenate([cell_metadata['source_ids'], cell_metadata['target_ids']])
        obs_df.index = cell_ids
    else:
        obs_df.index = [f'cell_{i}' for i in range(len(combined_latent))]
    
    # Create AnnData object
    adata = sc.AnnData(X=combined_latent, obs=obs_df)
    
    # Get cluster column to use
    cluster_column = None
    for col in ['leiden', 'cell_type', 'cluster', 'louvain']:
        if col in cell_metadata['source_adata'].obs.columns and col in cell_metadata['target_adata'].obs.columns:
            cluster_column = col
            break
    
    if not cluster_column:
        print("No cluster information found in both source and target datasets")
        return
    
    # Add cluster information
    try:
        source_ids = cell_metadata['source_ids']
        target_ids = cell_metadata['target_ids']
        
        source_adata = cell_metadata['source_adata']
        target_adata = cell_metadata['target_adata']
        
        # Get cluster labels
        source_clusters = source_adata.obs[cluster_column].loc[source_ids].values
        target_clusters = target_adata.obs[cluster_column].loc[target_ids].values
        
        # Combine clusters
        all_clusters = np.concatenate([source_clusters, target_clusters])
        adata.obs['cluster'] = all_clusters
        
        print(f"Successfully matched clusters for {len(source_ids)} source cells and {len(target_ids)} target cells")
    except Exception as e:
        print(f"Error matching clusters: {e}")
        return
    
    # Run UMAP
    print("Running UMAP...")
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=30, method='umap')
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=42, 
               init_pos='random', n_components=2)
    
    # Set scanpy plotting parameters
    sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(8, 8))
    sc.settings.figdir = output_dir
    
    # Create separate AnnData objects for source and target
    n_source = len(shared_latent_source)
    adata_source = adata[:n_source].copy()
    adata_target = adata[n_source:].copy()
    
    # Plot source dataset clusters
    print("Generating source dataset cluster UMAP...")
    sc.pl.umap(adata_source, color='cluster',
               title='',
               palette='tab20', size=10, alpha=0.7, 
               legend_loc='right margin',
               frameon=False,
               save=f"source_latent_umap_3.png")
    
    # Plot target dataset clusters
    print("Generating target dataset cluster UMAP...")
    sc.pl.umap(adata_target, color='cluster',
               title='',
               palette='tab20', size=10, alpha=0.7,
               legend_loc='right margin',
               frameon=False,
               save=f"target_latent_umap_3.png")
    
    # Plot joint clusters
    print("Generating joint cluster visualization...")
    sc.pl.umap(adata, color='cluster',
               title='',
               palette='tab20', size=10, alpha=0.7,
               legend_loc='right margin',
               frameon=False,
               save=f"joint_clusters{title_suffix.replace(' ', '_')}.png")
    
    print(f"Cluster UMAP visualizations saved to {output_dir}")
    
    return {
        'umap': adata.obsm['X_umap'],
        'n_source': n_source,
        'adata': adata
    }


def compute_cluster_consistency_loss(z_source, z_target, source_labels, target_labels):
    """Compute loss to maintain cluster relationships."""
    # Get unique labels across both datasets
    all_labels = torch.cat([source_labels, target_labels])
    unique_labels = torch.unique(all_labels)
    n_clusters = len(unique_labels)
    
    # Map original labels to indices for one-hot encoding
    label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
    source_indices = torch.tensor([label_to_idx[label.item()] for label in source_labels], 
                                device=z_source.device)
    target_indices = torch.tensor([label_to_idx[label.item()] for label in target_labels], 
                                device=z_target.device)
    
    # Create one-hot encodings
    source_onehot = F.one_hot(source_indices, n_clusters).float()
    target_onehot = F.one_hot(target_indices, n_clusters).float()
    
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
    X_source, X_target, Y_target, cell_metadata, decoder_model = load_data(
        config['source_model_path'],
        config['target_model_path'],
        config['decoder_model_path'],
        config['source_adata_path'],
        config['target_adata_path'],
        config['mapping_path']
    )
    
    # Create dataset with cell IDs for cluster matching
    source_cell_ids_np = np.array(cell_metadata['source_ids'])
    target_cell_ids_np = np.array(cell_metadata['target_ids'])
    
    # Create dictionaries for cell name to index mapping 
    source_adata = cell_metadata['source_adata']
    target_adata = cell_metadata['target_adata']
    source_cell_to_idx = {cell: idx for idx, cell in enumerate(source_adata.obs_names)}
    target_cell_to_idx = {cell: idx for idx, cell in enumerate(target_adata.obs_names)}
    
    # Find cluster column to use
    cluster_column = None
    for col in ['leiden', 'cell_type', 'cluster', 'louvain']:
        if col in source_adata.obs.columns and col in target_adata.obs.columns:
            cluster_column = col
            break
    
    # Get label encodings for clusters if available
    source_cluster_map = None
    target_cluster_map = None
    if cluster_column:
        if pd.api.types.is_categorical_dtype(source_adata.obs[cluster_column]):
            source_cluster_map = {cell: code for cell, code in 
                                zip(source_adata.obs_names, source_adata.obs[cluster_column].cat.codes)}
            target_cluster_map = {cell: code for cell, code in 
                                zip(target_adata.obs_names, target_adata.obs[cluster_column].cat.codes)}
        else:
            # Create a mapping from string values to integer codes
            unique_clusters = set(source_adata.obs[cluster_column]) | set(target_adata.obs[cluster_column])
            cluster_to_code = {cluster: i for i, cluster in enumerate(unique_clusters)}
            source_cluster_map = {cell: cluster_to_code[cluster] for cell, cluster in 
                               zip(source_adata.obs_names, source_adata.obs[cluster_column])}
            target_cluster_map = {cell: cluster_to_code[cluster] for cell, cluster in 
                               zip(target_adata.obs_names, target_adata.obs[cluster_column])}
    
    # Custom dataset for handling cell IDs and clusters
    class CellDataset(torch.utils.data.Dataset):
        def __init__(self, source_data, target_data, target_expression, source_ids, target_ids, 
                     source_cluster_map=None, target_cluster_map=None):
            self.source_data = torch.FloatTensor(source_data)
            self.target_data = torch.FloatTensor(target_data)
            self.target_expression = torch.FloatTensor(target_expression)
            self.source_ids = source_ids
            self.target_ids = target_ids
            self.source_cluster_map = source_cluster_map
            self.target_cluster_map = target_cluster_map
            
        def __len__(self):
            return len(self.source_data)
        
        def __getitem__(self, idx):
            source_id = self.source_ids[idx]
            target_id = self.target_ids[idx]
            
            # Add cluster information if available
            source_cluster = torch.tensor(-1, dtype=torch.long)  # Default if not available
            target_cluster = torch.tensor(-1, dtype=torch.long)  # Default if not available
            
            if self.source_cluster_map and source_id in self.source_cluster_map:
                source_cluster = torch.tensor(self.source_cluster_map[source_id], dtype=torch.long)
                
            if self.target_cluster_map and target_id in self.target_cluster_map:
                target_cluster = torch.tensor(self.target_cluster_map[target_id], dtype=torch.long)
            
            return (self.source_data[idx], self.target_data[idx], 
                    self.target_expression[idx], source_id, target_id,
                    source_cluster, target_cluster)
    
    dataset = CellDataset(X_source, X_target, Y_target, source_cell_ids_np, target_cell_ids_np,
                          source_cluster_map, target_cluster_map)
    
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        drop_last=False,
        num_workers=config.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        drop_last=False,
        num_workers=config.get('num_workers', 0)
    )
    
    # Initialize the model
    model = SCVISharedLatentEncoderWithScVIDecoder(
        source_dim=X_source.shape[1],
        target_dim=X_target.shape[1],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        target_scvi_model=decoder_model
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
    
    # For tracking losses during training
    loss_history = {
        'epoch': [],
        'mmd_loss': [],
        'recon_loss': [],
        'cluster_loss': [],
        'total_loss': [],
        'val_loss': []
    }
    
    # Train the model
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_losses = {
            'mmd': 0.0,
            'recon': 0.0,
            'cluster': 0.0,
            'total': 0.0
        }
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}") as pbar:
            for batch_data in train_loader:
                x_source, x_target, y_target, source_ids, target_ids, source_clusters, target_clusters = batch_data
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)
                source_clusters = source_clusters.to(device)
                target_clusters = target_clusters.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with reconstruction
                z_source, z_target, recon_source, recon_target = model(x_source, x_target)
                
                # Calculate losses
                mmd_loss_val = mmd_loss(z_source, z_target)
                recon_loss = F.mse_loss(recon_source, y_target) + F.mse_loss(recon_target, y_target)
                
                # Compute cluster loss if clusters are available
                cluster_loss = torch.tensor(0.0).to(device)
                if (source_clusters >= 0).any() and (target_clusters >= 0).any():
                    cluster_loss = compute_cluster_consistency_loss(z_source, z_target, source_clusters, target_clusters)
                
                # Combined loss
                loss = (config['loss_weights']['mmd'] * mmd_loss_val + 
                       config['loss_weights']['reconstruction'] * recon_loss +
                       config['loss_weights']['cluster'] * cluster_loss)
                
                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                
                # Update loss tracking
                epoch_losses['mmd'] += mmd_loss_val.item()
                epoch_losses['recon'] += recon_loss.item()
                epoch_losses['cluster'] += cluster_loss.item()
                epoch_losses['total'] += loss.item()
                
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        # Calculate average losses for the epoch
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                x_source, x_target, y_target, _, _, _, _ = batch_data
                x_source = x_source.to(device)
                x_target = x_target.to(device)
                
                z_source, z_target, _, _ = model(x_source, x_target)
                val_loss += mmd_loss(z_source, z_target).item()
        
        val_loss /= len(val_loader)
        
        # Update loss history
        loss_history['epoch'].append(epoch + 1)
        loss_history['mmd_loss'].append(epoch_losses['mmd'])
        loss_history['recon_loss'].append(epoch_losses['recon'])
        loss_history['cluster_loss'].append(epoch_losses['cluster'])
        loss_history['total_loss'].append(epoch_losses['total'])
        loss_history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}:")
        print(f"  MMD Loss: {epoch_losses['mmd']:.6f}")
        print(f"  Recon Loss: {epoch_losses['recon']:.6f}")
        print(f"  Cluster Loss: {epoch_losses['cluster']:.6f}")
        print(f"  Total Loss: {epoch_losses['total']:.6f}")
        print(f"  Validation Loss: {val_loss:.6f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{output_dir}/models/best_model_{run_id}.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
            early_stopping_counter = 0
            
            # Visualize the latent space for the best model
            if (epoch + 1) % config['visualize_every'] == 0:
                # Get all latent representations
                all_source_latent = []
                all_target_latent = []
                
                with torch.no_grad():
                    for batch_data in DataLoader(dataset, batch_size=config['batch_size']):
                        x_source, x_target, _, _, _, _, _ = batch_data
                        x_source = x_source.to(device)
                        x_target = x_target.to(device)
                        
                        z_source, z_target, _, _ = model(x_source, x_target)
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
        scheduler.step(val_loss)
        
        # Check for early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training and validation loss using matplotlib
    plt.figure(figsize=(12, 8))
    plt.plot(loss_history['epoch'], loss_history['mmd_loss'], label='MMD Loss')
    plt.plot(loss_history['epoch'], loss_history['recon_loss'], label='Reconstruction Loss')
    plt.plot(loss_history['epoch'], loss_history['cluster_loss'], label='Cluster Loss')
    plt.plot(loss_history['epoch'], loss_history['total_loss'], label='Total Loss')
    plt.plot(loss_history['epoch'], loss_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
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
        for batch_data in DataLoader(dataset, batch_size=config['batch_size']):
            x_source, x_target, _, _, _, _, _ = batch_data
            x_source = x_source.to(device)
            x_target = x_target.to(device)
            
            z_source, z_target, _, _ = model(x_source, x_target)
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
        'source_model_path': 'source_scvi_model/',
        'target_model_path': 'target_scvi_model/',
        'decoder_model_path': 'scvi-shared/decoder/models/scvi_decoder_20250417_164526',
        'source_adata_path': 'data/sq_cell_feature_1.h5ad',
        'target_adata_path': 'data/sq_cell_feature_2.h5ad',
        'mapping_path': 'data/cell_mapping.csv',
        
        # Model parameters
        'latent_dim': 10,       # Shared latent dimension
        'hidden_dim': 128,      # Hidden layer dimension for projection networks
        
        # Training parameters
        'batch_size': 1024,     # Batch size for training
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'train_split': 0.8,
        'early_stopping_patience': 15,
        'grad_clip': 5.0,
        'visualize_every': 5,   # Visualize latent space every N epochs
        'num_workers': 4,       # Number of workers for data loading
        
        # Loss weights
        'loss_weights': {
            'mmd': 1.0,          # Maximum Mean Discrepancy loss weight
            'reconstruction': 1.0, # Reconstruction loss weight
            'cluster': 0.2       # Cluster consistency loss weight
        },
        
        # Output directory
        'output_dir': 'scvi-shared/shared'
    }
    
    # Train the model
    model, run_id = train_shared_latent_model(config)
    
    print(f"\nTraining complete! Run ID: {run_id}")
    print(f"Results saved in {config['output_dir']}/results")
    print(f"Best model saved in {config['output_dir']}/models/best_model_{run_id}.pt")

