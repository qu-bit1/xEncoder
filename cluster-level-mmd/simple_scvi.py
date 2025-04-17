import os
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
from datetime import datetime
import torch

def visualize_shared_latent_space(source_adata_path, target_adata_path, output_dir):
    """
    Load pre-trained encoders and visualize the shared latent space.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    print("Loading data...")
    source_adata = sc.read_h5ad(source_adata_path)
    target_adata = sc.read_h5ad(target_adata_path)
    
    # Load latent representations from .pt files
    print("Loading latent representations...")
    source_latent = torch.load("source_latent_representation.pt").numpy()
    target_latent = torch.load("target_latent_representation.pt").numpy()
    
    # Add latent representations to AnnData objects
    source_adata.obsm['X_scVI'] = source_latent
    target_adata.obsm['X_scVI'] = target_latent
    
    # Combine latent representations
    combined_latent = np.vstack([source_latent, target_latent])
    
    # Create combined AnnData for visualization
    print("Creating visualization data...")
    dataset_labels = np.array(['Source'] * len(source_latent) + ['Target'] * len(target_latent))
    
    # Create metadata
    obs_df = pd.DataFrame({
        'dataset': dataset_labels
    })
    
    # Add cell IDs
    obs_df.index = np.concatenate([source_adata.obs_names, target_adata.obs_names])
    
    # Create AnnData object
    adata = sc.AnnData(X=combined_latent, obs=obs_df)
    adata.obsm['X_scVI'] = combined_latent  # Add latent representation to combined AnnData
    
    # Add cluster information if available
    cluster_column = None
    for col in ['leiden', 'cell_type', 'cluster', 'louvain']:
        if col in source_adata.obs.columns and col in target_adata.obs.columns:
            cluster_column = col
            break
    
    if cluster_column:
        source_clusters = source_adata.obs[cluster_column].values
        target_clusters = target_adata.obs[cluster_column].values
        adata.obs['cluster'] = np.concatenate([source_clusters, target_clusters])
    
    # Run UMAP with exact same parameters as train_source.py
    print("Running UMAP...")
    sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=30, method='umap')
    sc.tl.umap(adata, min_dist=0.25)  # Exact same min_dist as train_source.py
    
    # Set scanpy plotting parameters
    sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(8, 8))
    sc.settings.figdir = output_dir
    
    # Create separate AnnData objects for source and target
    n_source = len(source_latent)
    adata_source = adata[:n_source].copy()
    adata_target = adata[n_source:].copy()
    
    # Plot source clusters
    print("Generating source cluster visualization...")
    sc.pl.umap(adata_source, color='cluster',
               title='',
               palette='tab20', size=10, alpha=0.7,
               legend_loc='right margin',
               frameon=False,
               save=f"source_latent_umap_3.png")
    
    # Plot target clusters
    print("Generating target cluster visualization...")
    sc.pl.umap(adata_target, color='cluster',
               title='',
               palette='tab20', size=10, alpha=0.7,
               legend_loc='right margin',
               frameon=False,
               save=f"target_latent_umap_3.png")
    
    # Plot shared clusters with dataset markers
    print("Generating shared cluster visualization...")
    # Create a new column combining cluster and dataset information
    adata.obs['cluster_dataset'] = [
        f"{cluster} ({dataset})" 
        for cluster, dataset in zip(adata.obs['cluster'], adata.obs['dataset'])
    ]
    
    sc.pl.umap(adata, color='cluster_dataset',
               title='',
               palette='tab20', size=10, alpha=0.7,
               legend_loc='right margin',
               frameon=False,
               save=f"shared_clusters_{run_id}.png")
    
    print(f"\nVisualization complete! Run ID: {run_id}")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    # Configuration
    config = {
        'source_adata_path': 'source_latent_representation.h5ad',
        'target_adata_path': 'target_latent_representation.h5ad',
        'output_dir': 'scvi-shared/simple'
    }
    
    # Visualize the shared latent space
    visualize_shared_latent_space(
        config['source_adata_path'],
        config['target_adata_path'],
        config['output_dir']
    ) 
