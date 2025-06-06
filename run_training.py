#!/usr/bin/env python3
"""
Simple training script for T2 MSE version
Uses the optimized hyperparameters from hyperparameter tuning
"""

import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import load_data, create_rolling_windows
from src.train import train_model, save_model_checkpoint
from src.config import DEFAULT_CONFIG, TRAINING_CONFIG, setup_logging, OUTPUT_DIR

def main():
    """Main training function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting T2 MSE training with optimized hyperparameters")
    
    try:
        # Load data
        logger.info("Loading data...")
        t60_df, t2_df, factor_names, dates = load_data()
        
        # Create rolling windows
        logger.info("Creating rolling windows...")
        all_windows = create_rolling_windows(t60_df, t2_df, window_size=60)
        
        # Use the latest training window (most recent 60 months)
        latest_window = all_windows[-1]
        train_X, train_y, test_X, test_y, test_date = latest_window
        
        logger.info(f"Training on data ending {test_date}")
        logger.info(f"Training set: {train_X.shape}")
        logger.info(f"Test set: {test_X.shape}")
        
        # Combine default config with training config (includes optimized hyperparameters)
        config = DEFAULT_CONFIG.copy()
        config.update(TRAINING_CONFIG)
        
        logger.info("Training configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        
        # Train the model (use test data as validation for this single training run)
        logger.info("Starting model training...")
        model, history = train_model(train_X, train_y, test_X, test_y, config)
        
        # Save the trained model
        model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        save_model_checkpoint(model, model_path, config, history)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final validation loss: {history['best_val_loss']:.6f}")
        logger.info(f"Final validation return: {history['best_val_return']:.4f}")
        logger.info(f"Training time: {history['total_time']:.2f}s")
        logger.info(f"Total epochs: {history['final_epoch']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 