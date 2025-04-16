# Data Analysis Summary

## Dataset Overview

The analysis involves two main datasets:
1. A merged aligned cells dataset stored in `merged_aligned_cells.parquet`
2. Two single-cell datasets stored in H5AD format:
   - `sq_cell_feature_1.h5ad` (245,754 cells × 289 features)
   - `sq_cell_feature_2.h5ad` (271,693 cells × 5,001 features)

## Merged Aligned Cells Dataset

### Key Features
The dataset contains 31 columns with the following key features:
- Cell identification: `cell_id_source`, `cell_id_target`, `common_cell_id`
- Spatial coordinates: 
  - Source: `x_centroid_source`, `y_centroid_source`
  - Target: `x_centroid_target`, `y_centroid_target`
  - Transformed: `transformed_x_centroid`, `transformed_y_centroid`
- Count metrics:
  - Transcript counts
  - Control probe counts
  - Genomic control counts
  - Codeword counts (control, unassigned, deprecated)
  - Total counts
- Cell characteristics:
  - Cell area
  - Nucleus area
  - Nucleus count
  - Segmentation method

### Dataset Columns
```
Index(['cell_id_source', 'x_centroid_source', 'y_centroid_source',
       'transcript_counts_source', 'control_probe_counts_source',
       'genomic_control_counts_source', 'control_codeword_counts_source',
       'unassigned_codeword_counts_source',
       'deprecated_codeword_counts_source', 'total_counts_source',
       'cell_area_source', 'nucleus_area_source', 'nucleus_count_source',
       'segmentation_method_source', 'cell_id_target', 'x_centroid_target',
       'y_centroid_target', 'transcript_counts_target',
       'control_probe_counts_target', 'genomic_control_counts_target',
       'control_codeword_counts_target', 'unassigned_codeword_counts_target',
       'deprecated_codeword_counts_target', 'total_counts_target',
       'cell_area_target', 'nucleus_area_target', 'nucleus_count_target',
       'segmentation_method_target', 'transformed_x_centroid',
       'transformed_y_centroid', 'common_cell_id'], dtype='object')
```

### Important Observations
1. There are 173,745 unique `common_cell_id` values, indicating the number of unique cell mappings
2. The data shows many-to-one mapping patterns, likely due to a nearest neighbor approach in cell alignment
3. Example mapping patterns:
   - Multiple target cells can map to the same source cell
   - The same source cell can have multiple target cell mappings

## Preprocessed Spatially aligned Datasets
Contains common cell id in both to map cells from both datasets.
### Dataset 1 (sq_cell_feature_1.h5ad)
- Dimensions: 245,754 cells × 289 features
- Contains:
  - Cell metadata (23 features)
  - Gene expression data
  - Spatial coordinates
  - Dimensionality reduction results (PCA, UMAP)
  - Clustering information (Leiden)

### Dataset 2 (sq_cell_feature_2.h5ad)
- Dimensions: 271,693 cells × 5,001 features
- Contains similar structure to Dataset 1 but with:
  - More cells
  - Significantly more features (5,001 vs 289)
  - Same metadata structure

## Analysis Notes
- The data appears to be from a spatial transcriptomics experiment
- Cell alignment between datasets uses a transformation approach
- The many-to-one mapping suggests potential challenges in cell matching accuracy
- Both datasets include comprehensive cell metadata and gene expression information 