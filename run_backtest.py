#!/usr/bin/env python3
"""
Run complete backtest for T2 MSE version
Uses the optimized hyperparameters across the entire historical period
"""

import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.backtest import run_backtest
from src.config import setup_logging

def main():
    """Main backtest function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting complete T2 MSE backtest with optimized hyperparameters")
    
    try:
        # Run the complete backtest
        # parallel=False to avoid MPS/CPU device mismatch issues
        # save_forecasts=True will save the enhanced T60 file
        results_df, analysis = run_backtest(
            data_dir='data/',
            parallel=False,  # Use sequential processing to avoid device issues
            start_date=None,  # Use all available data
            end_date=None,    # Use all available data
            config=None,      # Use optimized config from hyperparameter tuning
            save_results=True,  # Save results to CSV files
            save_forecasts=True  # Save enhanced T60 forecasts
        )
        
        logger.info("Backtest completed successfully!")
        
        # Print key results
        if analysis:
            logger.info("=== BACKTEST RESULTS SUMMARY ===")
            logger.info(f"Total months processed: {analysis.get('n_months', 'N/A')}")
            logger.info(f"Average monthly return: {analysis.get('avg_monthly_return', 0):.4f}")
            logger.info(f"Annual return: {analysis.get('annual_return', 0):.2%}")
            logger.info(f"Annual volatility: {analysis.get('annual_volatility', 0):.2%}")
            logger.info(f"Sharpe ratio: {analysis.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Average hit rate: {analysis.get('avg_hit_rate', 0):.3f}")
            logger.info(f"Win rate: {analysis.get('win_rate', 0):.3f}")
            logger.info(f"Max drawdown: {analysis.get('max_drawdown', 0):.2%}")
            logger.info(f"Total backtest time: {analysis.get('total_backtest_time', 0):.2f}s")
            
            logger.info("Results saved to:")
            logger.info("  - outputs/backtest_MSE_results.csv (detailed results)")
            logger.info("  - outputs/backtest_MSE_analysis.csv (performance summary)")
            logger.info("  - outputs/T60_Enhanced_MSE.xlsx (enhanced forecasts)")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main() 