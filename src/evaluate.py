import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import pandas as pd

# Import project modules
from models.multi_encoder import MultiEncoderAutoencoder
from utils.analysis import (
    calculate_gene_correlations,
    analyze_encoder_alignment,
    visualize_latent_space
)

def evaluate_model(model_path, config):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the saved model
        config: Dictionary containing evaluation configuration
    """
    # Create directories for results
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # Load data
    X_source = np.load(config['data_paths']['source_data'])
    X_target = np.load(config['data_paths']['target_data'])
    
    try:
        target_genes = np.load(config['data_paths']['target_genes'])
    except:
        print("Gene name file not found. Proceeding without gene names.")
        target_genes = np.array([f"gene_{i}" for i in range(X_target.shape[1])])
    
    # Create dataset and data loader
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_source), 
        torch.FloatTensor(X_target)
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False
    )
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    source_dim = X_source.shape[1]
    target_dim = X_target.shape[1]
    
    model = MultiEncoderAutoencoder(
        source_dim=source_dim,
        target_dim=target_dim,
        latent_dim=config['model']['latent_dim'],
        hidden_dim=config['model']['hidden_dim']
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get run ID from model path
    run_id = os.path.basename(model_path).split('_')[-1].split('.')[0]
    
    # Perform evaluation
    print("\n==== MODEL EVALUATION ====")
    
    # Get predictions and latent representations
    all_predictions = []
    all_targets = []
    all_latents = []
    
    with torch.no_grad():
        for x_source, x_target in data_loader:
            x_source, x_target = x_source.to(device), x_target.to(device)
            
            # Get predictions using only the source encoder
            recon, z_source = model(x_source)
            
            all_predictions.append(recon.cpu().numpy())
            all_targets.append(x_target.cpu().numpy())
            all_latents.append(z_source.cpu().numpy())
    
    # Concatenate results
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    latents = np.vstack(all_latents)
    
    # Save the results
    np.save(f'results/metrics/predicted_target_data_{run_id}.npy', predictions)
    np.save(f'results/metrics/true_target_data_{run_id}.npy', targets)
    np.save(f'results/metrics/latent_representations_{run_id}.npy', latents)
    
    # Overall metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    r2 = r2_score(targets.flatten(), predictions.flatten())
    corr, _ = pearsonr(targets.flatten(), predictions.flatten())
    
    # Convert numpy types to native Python types for JSON serialization
    eval_results = {
        "mse": float(mse),
        "r2": float(r2),
        "pearson_correlation": float(corr),
        "run_id": run_id
    }
    
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson Correlation: {corr:.4f}")
    
    # Save evaluation results
    with open(f'results/metrics/evaluation_results_{run_id}.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    # 1. Latent Space Analysis
    print("\n==== LATENT SPACE ANALYSIS ====")
    
    # t-SNE visualization of final latent space
    latent_2d = visualize_latent_space(latents, title=f"Final Latent Space t-SNE {run_id}")
    
    # Latent feature distribution (showing first 16 dimensions)
    plt.figure(figsize=(15, 10))
    for i in range(min(16, latents.shape[1])):
        plt.subplot(4, 4, i+1)
        sns.histplot(latents[:, i], kde=True)
        plt.title(f"Dimension {i+1}")
    plt.tight_layout()
    plt.savefig(f"results/figures/final_latent_dimensions_{run_id}.png", dpi=300)
    plt.close()
    
    # 2. Per-gene correlation analysis
    print("\n==== PER-GENE CORRELATION ANALYSIS ====")
    
    gene_corrs, _, _ = calculate_gene_correlations(model, data_loader, device, target_dim)
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(gene_corrs, kde=True)
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Count')
    plt.title('Distribution of Gene-Level Correlations')
    plt.savefig(f"results/figures/final_gene_correlation_distribution_{run_id}.png", dpi=300)
    plt.close()
    
    # Find best and worst predicted genes
    top_genes_idx = np.argsort(gene_corrs)[-20:]
    bottom_genes_idx = np.argsort(gene_corrs)[:20]
    
    print("Top 5 Best Predicted Genes:")
    for idx in reversed(top_genes_idx[-5:]):
        print(f"{target_genes[idx]}: {gene_corrs[idx]:.4f}")
    
    print("\nBottom 5 Worst Predicted Genes:")
    for idx in bottom_genes_idx[:5]:
        print(f"{target_genes[idx]}: {gene_corrs[idx]:.4f}")
    
    # Save complete gene correlation results
    gene_corr_df = pd.DataFrame({
        'gene': target_genes,
        'correlation': gene_corrs
    })
    gene_corr_df.to_csv(f'results/metrics/gene_correlations_{run_id}.csv', index=False)
    
    # 3. Random sample predictions visualization
    print("\n==== SAMPLE PREDICTIONS ====")
    
    # Plot predictions vs actual for random subset of samples
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    random_indices = np.random.choice(len(targets), size=min(9, len(targets)), replace=False)
    
    for i, idx in enumerate(random_indices[:9]):
        true_values = targets[idx]
        pred_values = predictions[idx]
        
        # Sample 1000 points if there are too many genes
        if len(true_values) > 1000:
            sample_idx = np.random.choice(len(true_values), 1000, replace=False)
            true_sample = true_values[sample_idx]
            pred_sample = pred_values[sample_idx]
        else:
            true_sample = true_values
            pred_sample = pred_values
        
        axes[i].scatter(true_sample, pred_sample, alpha=0.5, s=3)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        
        # Add correlation to the subplot
        sample_corr, _ = pearsonr(true_values, pred_values)
        axes[i].set_title(f'Sample {idx}, r={sample_corr:.3f}')
        
        # Add diagonal line
        min_val = min(true_sample.min(), pred_sample.min())
        max_val = max(true_sample.max(), pred_sample.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"results/figures/final_sample_predictions_{run_id}.png", dpi=300)
    plt.close()
    
    return eval_results

if __name__ == "__main__":
    # Default configuration
    config = {
        'data_paths': {
            'source_data': 'data/processed/source_data.npy',
            'target_data': 'data/processed/target_data.npy',
            'source_genes': 'data/processed/source_genes.npy',
            'target_genes': 'data/processed/target_genes.npy'
        },
        'model': {
            'latent_dim': 384,
            'hidden_dim': 128
        },
        'training': {
            'batch_size': 128
        }
    }
    
    # Evaluate the model (replace with your model path)
    model_path = 'models/checkpoints/best_model_XXXXXXXX_XXXXXX.pt'  # Update this
    evaluate_model(model_path, config)