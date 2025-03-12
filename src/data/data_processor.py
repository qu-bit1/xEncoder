import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import anndata

def prepare_data(source_path, target_path, alignment_path, output_dir='data/processed'):
    """
    Process and prepare the data for training.
    
    Args:
        source_path: Path to the source dataset (h5 file)
        target_path: Path to the target dataset (h5 file)
        alignment_path: Path to the alignment information (parquet file)
        output_dir: Directory to save processed data
    """
    # Load the datasets
    df = pd.read_parquet(alignment_path)
    adata_source = sc.read_10x_h5(source_path)
    adata_target = sc.read_10x_h5(target_path)

    print(f"Source dataset shape: {adata_source.shape}")
    print(f"Target dataset shape: {adata_target.shape}")
    print(f"Merged alignment dataframe shape: {df.shape}")

    # Extract the common cell IDs that are present in both datasets
    common_cells = df['common_cell_id'].dropna().unique()
    print(f"Number of common cells: {len(common_cells)}")

    # First, make sure the cell IDs in the AnnData objects match the format in the alignment dataframe
    # This might require parsing or reformatting depending on your specific cell ID formats
    source_cells = df['cell_id_source'].values
    target_cells = df['cell_id_target'].values

    # Create a mapping dictionary between source and target cell IDs
    cell_mapping = dict(zip(source_cells, target_cells))

    # Now, let's filter the AnnData objects to only include common cells
    # We'll need to make sure cell IDs are in the same format as the obs index
    source_cells_in_adata = [cell for cell in source_cells if cell in adata_source.obs_names]
    target_cells_in_adata = [cell for cell in target_cells if cell in adata_target.obs_names]

    print(f"Source cells found in adata_source: {len(source_cells_in_adata)}")
    print(f"Target cells found in adata_target: {len(target_cells_in_adata)}")

    # Filter AnnData objects to only include matched cells
    adata_source_filtered = adata_source[adata_source.obs_names.isin(source_cells_in_adata)].copy()
    adata_target_filtered = adata_target[adata_target.obs_names.isin(target_cells_in_adata)].copy()

    print(f"Filtered source dataset: {adata_source_filtered.shape}")
    print(f"Filtered target dataset: {adata_target_filtered.shape}")

    # Make sure the cells are in the same order in both datasets
    # Create dataframes with the filtered data
    source_order = pd.DataFrame(index=adata_source_filtered.obs_names)
    source_order['target_id'] = source_order.index.map(lambda x: cell_mapping.get(x))

    # Reorder both AnnData objects to ensure they match cell-by-cell
    source_order = source_order.dropna()  # Remove any cells that don't have a mapping
    adata_source_final = adata_source_filtered[adata_source_filtered.obs_names.isin(source_order.index)].copy()
    adata_target_final = adata_target_filtered[adata_target_filtered.obs_names.isin(source_order['target_id'])].copy()

    # Reorder adata_target_final to match the order of cells in adata_source_final
    adata_source_final = adata_source_final[source_order.index].copy()
    adata_target_final = adata_target_final[source_order['target_id']].copy()

    print(f"Final source dataset: {adata_source_final.shape}")
    print(f"Final target dataset: {adata_target_final.shape}")

    # Normalize the data
    sc.pp.normalize_total(adata_source_final, target_sum=1e4)
    sc.pp.normalize_total(adata_target_final, target_sum=1e4)
    sc.pp.log1p(adata_source_final)
    sc.pp.log1p(adata_target_final)

    # Get the gene names from both datasets
    source_genes = adata_source_final.var_names.tolist()
    target_genes = adata_target_final.var_names.tolist()

    print(f"Number of genes in source dataset: {len(source_genes)}")
    print(f"Number of genes in target dataset: {len(target_genes)}")

    # Check for common genes
    common_genes = set(source_genes).intersection(set(target_genes))
    source_unique_genes = set(source_genes) - common_genes
    target_unique_genes = set(target_genes) - common_genes

    print(f"Number of common genes: {len(common_genes)}")
    print(f"Number of unique genes in source: {len(source_unique_genes)}")
    print(f"Number of unique genes in target: {len(target_unique_genes)}")

    # Convert to numpy arrays for use in PyTorch or TensorFlow
    X_source = adata_source_final.X.toarray() if sp.issparse(adata_source_final.X) else adata_source_final.X
    X_target = adata_target_final.X.toarray() if sp.issparse(adata_target_final.X) else adata_target_final.X

    # Save processed data for the autoencoder
    np.save(f'{output_dir}/source_data.npy', X_source)
    np.save(f'{output_dir}/target_data.npy', X_target)

    # Save gene names for reference
    np.save(f'{output_dir}/source_genes.npy', np.array(source_genes))
    np.save(f'{output_dir}/target_genes.npy', np.array(target_genes))

    # Also save the original AnnData objects if needed for downstream analysis
    adata_source_final.write(f'{output_dir}/processed_source_data.h5ad')
    adata_target_final.write(f'{output_dir}/processed_target_data.h5ad')

    print("Data preparation complete. Final datasets are ready for the autoencoder model.")
    
    return {
        'source_data': X_source,
        'target_data': X_target,
        'source_genes': source_genes,
        'target_genes': target_genes
    }