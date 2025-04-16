import scvi
import scanpy as sc
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set precision for better GPU performance on Tensor Cores
torch.set_float32_matmul_precision('high')

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA available — using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available — using CPU.")

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# Load source data
adata_source = sc.read_h5ad("data/sq_cell_feature_1.h5ad")

# Setup for scVI
scvi.model.SCVI.setup_anndata(adata_source)

# Initialize model
model = scvi.model.SCVI(
    adata_source,
    n_layers=1,
    n_latent=10,
    gene_likelihood="nb",
    dispersion="gene-cell"
)

# Train model and collect history
history = model.train(
    max_epochs=100,
    early_stopping=True,
    early_stopping_patience=20,
    accelerator=accelerator,
    devices=1,
    batch_size=1024,
)

history = model.history

# Plot training loss
plt.figure(figsize=(6, 4))
plt.plot(history["elbo_train"], label="Train ELBO")
plt.plot(history["elbo_validation"], label="Validation ELBO")
plt.xlabel("Epoch")
plt.ylabel("ELBO Loss")
plt.title("source Dataset — Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("source_training_loss_3.png", dpi=200)
plt.close()

# Get latent representation
adata_source.obsm["X_scVI"] = model.get_latent_representation()

# Clustering & UMAP
sc.pp.neighbors(adata_source, use_rep="X_scVI")
sc.tl.leiden(adata_source, resolution=0.5)
sc.tl.umap(adata_source, min_dist=0.25)

sc.pl.umap(
    adata_source,
    color=["leiden"],
    title="source Dataset — scVI Latent Space",
    frameon=False,
    save="source_latent_umap_3.png"
)

# Save everything
model.save("source_scvi_model", overwrite=True)
np.save("source_latent_representation.npy", adata_source.obsm["X_scVI"])
torch.save(torch.tensor(adata_source.obsm["X_scVI"]), "source_latent_representation.pt")
adata_source.write("source_latent_representation.h5ad")

