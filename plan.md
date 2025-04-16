# plan

## Problem Statement
This project addresses a spatial transcriptomics problem where we have two datasets from same tissue. One has less number of genes than the other for
about the same number of cells. The task is to train a model such that when inferencing with the dataset with less number of genes it predicts the 
feature matrix of the larger dataset.

## Sub-problems
I've broken down this task to several sub-tasks as follows:
- [x] Align both the datasets spatially to get a cell mapping from both datasets.
  - [x] First align both the tissues using the code given in the directory `tissue-alignment`
  - Got the cell mapping with cells from source dataset mapping to many cells in the target dataset due to the nearest neighbor approach.
  - That one to many cell mapping is good as it may lead to some important biological insights.
- [x] Cell type annotations to cluster them.
- [x] Preprocess datasets. 
- Define a model architecture

### Model architecure
I've thought of an auto-encoder approach for this problem with the following specifics:
- Two seperate encoders for each datasets
- A common shared latent space to align the latent vectors from both the datasets.
- A single decoder to reconstruct the larger genes dataset with maybe a reconstruction loss with the original dataset.

From this I have currently tried and implemented a lot of encoders. My main criteria to see if the encoder was right was to see the latent space when
each dataset is passed through the encoder seperatly whether it forms cells clusters umap clearly or not. Following is the bried of all the implementions:

- `og_training`
  - contains deep model, simpe auto encoder, and a VAE model and its training & visualisation scripts and the results of runs with various parameters.
  - Didn't work as expected the clusters were not clearly seperated and were not accurate.
- `scvi_encoders`
  - Contains training scripts for source and target dataset. Uses the scVI library for the models.
  - Creates great seperated clusters in the latent space works great till now.

## Datasets summary
Data information and structures are mentioned in the `data_summary.md`.

## To-Do
- Create a pipeline using scVI encoders to a single shared latent space to visualise the UMAP of that latent space. Ideally clusters of same type of 
cells from both the seperate encoders should overlap in the latent space. Use appropriate losses like MMD.
- After successfully creating and testing the shared latent space create another scVI decoder to output the larger dataset. Add reconstruction loss.



