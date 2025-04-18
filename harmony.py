import scvi
import scanpy as sc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import harmonypy as hp
import os

# Set precision for better GPU performance
torch.set_float32_matmul_precision('high')

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA available — using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available — using CPU.")

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# Cell type annotations
cl_annotations1 = {
    '1': "Epithelial", '2': "Fibroblasts", '0': "T cells", '3': "Macrophages", '4': "Endothelial",
    '5': "Endothelial", '6': "Smooth muscles", '7': "Epithelial", '8': "Epithelial", '9': "B cells",
    '10': "Epithelial", '11': "Plasma cells", '12': "Epithelial"
}

cl_annotations2 = {
    '1': "Endothelial", '2': "Fibroblasts", '0': "T cells", '4': "Macrophages", '3': "Epithelial",
    '5': "Endothelial", '6': "Epithelial", '7': "Epithelial", '8': "B cells", '9': "Endothelial",
    '10': "Mast cells", '11': "Macrophages"
}

# 1. Process source data
print("Loading source data...")
if os.path.exists("source_latent_representation.h5ad") and os.path.exists("source_scvi_model/model.pt"):
    # Load pre-trained model and latent representation
    adata_source = sc.read_h5ad("source_latent_representation.h5ad")
    print("Successfully loaded source latent representation.")
else:
    print("Could not find source model or latent representation files.")
    exit(1)

# 2. Process target data
print("Loading target data...")
if os.path.exists("target_latent_representation.h5ad") and os.path.exists("target_scvi_model/model.pt"):
    # Load pre-trained model and latent representation
    adata_target = sc.read_h5ad("target_latent_representation.h5ad")
    print("Successfully loaded target latent representation.")
else:
    print("Could not find target model or latent representation files.")
    exit(1)

# 3. Add dataset labels and cell type annotations
print("Adding metadata and combining datasets...")
adata_source.obs['batch'] = 'source'
adata_target.obs['batch'] = 'target'

# Add cell type annotations
adata_source.obs['cell_type'] = adata_source.obs['leiden'].astype(str).map(cl_annotations1)
adata_target.obs['cell_type'] = adata_target.obs['leiden'].astype(str).map(cl_annotations2)

# Make a copy of the leiden clusters with dataset prefix for visualization
adata_source.obs['orig_clusters'] = 'S_' + adata_source.obs['leiden'].astype(str)
adata_target.obs['orig_clusters'] = 'T_' + adata_target.obs['leiden'].astype(str)

# 4. Combine the datasets
# Only keep necessary columns to avoid potential conflicts
keep_cols = ['batch', 'cell_type', 'leiden', 'orig_clusters']
common_cols = set(adata_source.obs.columns).intersection(set(keep_cols))
adata_source.obs = adata_source.obs[list(common_cols)]
common_cols = set(adata_target.obs.columns).intersection(set(keep_cols))
adata_target.obs = adata_target.obs[list(common_cols)]

# Combining datasets - focus only on metadata and latent space
adata_combined = sc.AnnData(
    obs=pd.concat([adata_source.obs, adata_target.obs], axis=0),
)

# Add latent representations to combined dataset
adata_combined.obsm["X_scVI"] = np.vstack([
    adata_source.obsm["X_scVI"],
    adata_target.obsm["X_scVI"]
])

# 5. Integration with Harmony
print("Integrating with Harmony...")
# Prepare data for Harmony
data_mat = adata_combined.obsm["X_scVI"]
meta_data = adata_combined.obs

# Run Harmony for batch correction
# The correct way to extract the embedding matrix:
# ho = hp.run_harmony(data_mat, meta_data, ['batch'], theta=[2], max_iter_harmony=20)
ho = hp.run_harmony(
    data_mat, 
    meta_data, 
    ['batch', 'cell_type'],  # Use both batch and cell type 
    theta=[2, 0.5],          # Lower theta for cell_type
    lamb=[1, 1],          # Equal weighting of variables
    max_iter_harmony=30
)
# Extract the corrected embedding matrix from the Harmony object
harmony_embeddings = ho.Z_corr.T  # Transpose to get cells in rows
adata_combined.obsm["X_harmony"] = harmony_embeddings

# 6. Generate UMAP and clusters on integrated space
print("Generating UMAP and clusters...")
sc.pp.neighbors(adata_combined, use_rep="X_harmony")
sc.tl.umap(adata_combined)
sc.tl.leiden(adata_combined, resolution=0.5, key_added='integrated_leiden')

# 7. Visualization
print("Creating visualizations...")
# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Plot by dataset origin
sc.pl.umap(
    adata_combined, 
    color=['batch'], 
    title="Integrated Space - Dataset Origin",
    frameon=False,
    save="_two_integrated_batch.png"
)

# Plot by original clusters
sc.pl.umap(
    adata_combined, 
    color=['orig_clusters'], 
    title="Integrated Space - Original Clusters",
    frameon=False,
    save="_two_integrated_orig_clusters.png"
)

# Plot by cell type
sc.pl.umap(
    adata_combined, 
    color=['cell_type'], 
    title="Integrated Space - Cell Types",
    frameon=False,
    save="_two_integrated_cell_types.png"
)

# Plot by integrated clusters
sc.pl.umap(
    adata_combined, 
    color=['integrated_leiden'], 
    title="Integrated Space - New Clusters",
    frameon=False,
    save="_two_integrated_leiden.png"
)

# 8. Save the integrated results
print("Saving results...")
adata_combined.write("two_integrated_representation.h5ad")

print("Integration pipeline completed!")