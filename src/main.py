import argparse
import json
import os
import shutil

from data.data_processor import prepare_data
from train import train_model
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Xenium Multi-Encoder Autoencoder')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='Path to configuration file')
    parser.add_argument('--prepare_data', action='store_true',
                        help='Prepare data for training')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Prepare data if requested
    if args.prepare_data:
        print("Preparing data...")
        
        # Create directories if they don't exist
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Copy raw data files to data/raw if they're not already there
        raw_files = [
            'cell_feature_matrix.h5',
            'cell_feature_matrix2.h5',
            'merged_aligned_cells.parquet'
        ]
        
        for file in raw_files:
            if os.path.exists(file) and not os.path.exists(f'data/raw/{file}'):
                shutil.copy(file, f'data/raw/{file}')
        
        # Process the data
        data = prepare_data(
            source_path='data/raw/cell_feature_matrix.h5',
            target_path='data/raw/cell_feature_matrix2.h5',
            alignment_path='data/raw/merged_aligned_cells.parquet',
            output_dir='data/processed'
        )
        
        print("Data preparation complete.")
    
    # Train the model if requested
    if args.train:
        print("Training model...")
        model, run_id = train_model(config)
        print(f"Training complete. Run ID: {run_id}")
    
    # Evaluate the model if requested
    if args.evaluate:
        if args.model_path is None:
            print("Error: Model path must be provided for evaluation.")
            return
        
        print(f"Evaluating model: {args.model_path}")
        results = evaluate_model(args.model_path, config)
        print("Evaluation complete.")
        print(f"Results: MSE={results['mse']:.4f}, R²={results['r2']:.4f}, Pearson={results['pearson_correlation']:.4f}")

if __name__ == "__main__":
    main()