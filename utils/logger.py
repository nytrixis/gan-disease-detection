"""
Logging utilities for training
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_training_params(logger, params):
    """
    Log training hyperparameters
    
    Args:
        logger: Logger instance
        params: Dictionary of parameters
    """
    logger.info("="*70)
    logger.info("Training Parameters:")
    logger.info("="*70)
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*70)

def log_epoch_metrics(logger, epoch, metrics):
    """
    Log metrics for an epoch
    
    Args:
        logger: Logger instance
        epoch: Epoch number
        metrics: Dictionary of metrics
    """
    logger.info(f"Epoch {epoch}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

class TrainingLogger:
    """Training logger with automatic file naming"""
    
    def __init__(self, experiment_name, log_dir='logs'):
        """
        Initialize training logger
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory for logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(experiment_name, log_file)
        
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_params(self, params):
        """Log training parameters"""
        log_training_params(self.logger, params)
    
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics"""
        log_epoch_metrics(self.logger, epoch, metrics)

if __name__ == '__main__':
    # Example usage
    logger = TrainingLogger('test_experiment')
    
    logger.log_params({
        'batch_size': 16,
        'learning_rate': 0.0001,
        'epochs': 50
    })
    
    logger.log_epoch(1, {
        'train_loss': 0.5432,
        'val_loss': 0.6123,
        'accuracy': 0.8567
    })
    
    logger.info("Training complete!")
