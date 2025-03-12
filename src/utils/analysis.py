import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

def calculate_gene_correlations(model, data_loader, device, num_genes=5001):
    """
    Calculate per-gene correlations between predicted and actual gene expression.
    
    Args:
        model: The trained model
        data_loader: DataLoader containing validation data
        device: Device to run the model on
        num_genes: Number of genes in the target dataset
        
    Returns:
        gene_correlations: List of correlation values for each gene
        all_targets: Array of actual gene expression values
        all_predictions: Array of predicted gene expression values
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for x_source, x_target in data_loader:
            x_source = x_source.to(device)
            x_target = x_target.to(device)
            
            z_source = model.encode_source(x_source)
            recon = model.decode(z_source)
            
            all_targets.append(x_target.cpu().numpy())
            all_predictions.append(recon.cpu().numpy())
    
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    gene_correlations = []
    for i in range(num_genes):
        corr, _ = pearsonr(all_targets[:, i], all_predictions[:, i])
        gene_correlations.append(corr)
    
    return gene_correlations, all_targets, all_predictions

def analyze_encoder_alignment(model, data_loader, device):
    """
    Analyze the alignment between source and target encoders.
    
    Args:
        model: The trained model
        data_loader: DataLoader containing validation data
        device: Device to run the model on
        
    Returns:
        avg_similarity: Average cosine similarity between source and target encodings
        source_latents: Array of source latent vectors
        target_latents: Array of target latent vectors
    """
    model.eval()
    source_latents = []
    target_latents = []
    
    with torch.no_grad():
        for x_source, x_target in data_loader:
            x_source = x_source.to(device)
            x_target = x_target.to(device)
            
            z_source = model.encode_source(x_source).cpu().numpy()
            z_target = model.encode_target(x_target).cpu().numpy()
            
            source_latents.append(z_source)
            target_latents.append(z_target)
    
    source_latents = np.vstack(source_latents)
    target_latents = np.vstack(target_latents)
    
    # Calculate average cosine similarity
    similarities = []
    for i in range(len(source_latents)):
        cos_sim = cosine_similarity([source_latents[i]], [target_latents[i]])[0][0]
        similarities.append(cos_sim)
    
    avg_similarity = np.mean(similarities)
    
    # Visualize source and target encodings
    combined = np.vstack([source_latents, target_latents])
    labels = np.array(['Source'] * len(source_latents) + ['Target'] * len(target_latents))
    
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=combined_2d[:, 0], y=combined_2d[:, 1], hue=labels)
    plt.title(f"Source vs Target Encodings (Avg. Cosine Similarity: {avg_similarity:.3f})")
    
    # Create directory if it doesn't exist
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/encoder_alignment.png", dpi=300)
    
    return avg_similarity, source_latents, target_latents

def visualize_latent_space(latent_vectors, labels=None, title="t-SNE Visualization of Latent Space"):
    """
    Visualize the latent space using t-SNE.
    
    Args:
        latent_vectors: Array of latent vectors
        labels: Optional array of labels for coloring points
        title: Title for the plot
        
    Returns:
        latent_2d: 2D t-SNE projection of latent vectors
    """
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=labels)
    else:
        sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1])
    plt.title(title)
    
    # Create directory if it doesn't exist
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/{title.lower().replace(' ', '_')}.png", dpi=300)
    
    return latent_2d

def noise_sensitivity_test(model, example_input, noise_levels, device):
    """
    Test the model's sensitivity to input noise.
    
    Args:
        model: The trained model
        example_input: Example input tensor
        noise_levels: List of noise standard deviations to test
        device: Device to run the model on
        
    Returns:
        results: List of tuples (noise_level, reconstruction_error, latent_error)
    """
    model.eval()
    results = []
    
    original_input = example_input.clone()
    original_z = model.encode_source(original_input.to(device))
    original_output = model.decode(original_z)
    
    for noise in noise_levels:
        # Add noise to input
        noisy_input = original_input + torch.randn_like(original_input) * noise
        noisy_input = torch.clamp(noisy_input, 0, None)  # Ensure non-negative if needed
        
        # Encode and decode
        with torch.no_grad():
            z = model.encode_source(noisy_input.to(device))
            output = model.decode(z)
            
        # Calculate metrics
        recon_error = torch.nn.MSELoss()(output, original_output).item()
        latent_error = torch.nn.MSELoss()(z, original_z).item()
        
        results.append((noise, recon_error, latent_error))
    
    return results