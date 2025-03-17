# Tissue Alignment
Before moving forward we need to align both the tissues to get an approximate cell mapping between the two tissues.
install dependencies as per [STalign](https://github.com/JEFworks-Lab/STalign).

## Files
Data files obtained from Xenium
- `cells.parquet`, `cells2.parquet`: Cell data in Parquet format

See the notebook for all the info

- `stalign_fin.ipynb`: Notebook for alignment using LDDMM
- `point_annotator.py`: Interactive tool for annotating corresponding points between two images. Takes two .npz image files as input and allows users to mark matching points. \\

- `Rep1.npz`, `Rep2.npz`: Input image data files generated from the notebook
- `Rep1_points.npy`, `Rep2_points.npy`: Saved point annotations for the respective images

Output file - `merged_aligned_cells.parquet`

## Usage

To annotate corresponding points between two images:

```bash
python point_annotator.py image1.npz image2.npz
```

The tool will:
1. Display both images side by side
2. Allow you to name landmark structures
3. Let you click alternating points between source and target images
4. Save the point correspondences as .npy files

Points are saved automatically when you press Enter without entering a new landmark name.

## STalign
To further align the tissue we're using [STalign](https://github.com/JEFworks-Lab/STalign). STalign solves a mapping that minimizes the dissimilarity between a source and a target ST dataset subject to regularization penalties.