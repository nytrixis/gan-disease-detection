"""
Model checkpointing utilities
"""

import torch
from pathlib import Path
import json
from datetime import datetime

class ModelCheckpoint:
    """
    Save model checkpoints during training
    """
    
    def __init__(self, checkpoint_dir, mode='max', monitor='val_f1', 
                 save_best_only=True, verbose=True):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            mode: 'max' or 'min' for monitoring metric
            monitor: Metric to monitor
            save_best_only: Only save when metric improves
            verbose: Print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.mode = mode
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.history = []
    
    def is_better(self, current_value):
        """Check if current value is better than best"""
        if self.mode == 'max':
            return current_value > self.best_value
        else:
            return current_value < self.best_value
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            filename: Custom filename (optional)
        """
        current_value = metrics.get(self.monitor, None)
        
        if current_value is None:
            if self.verbose:
                print(f"⚠️  Warning: {self.monitor} not found in metrics")
            return False
        
        # Check if should save
        should_save = True
        if self.save_best_only:
            should_save = self.is_better(current_value)
        
        if should_save:
            if filename is None:
                if self.save_best_only:
                    filename = 'best_model.pth'
                else:
                    filename = f'checkpoint_epoch_{epoch:03d}.pth'
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # Prepare checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save
            torch.save(checkpoint, checkpoint_path)
            
            # Update best value
            if self.is_better(current_value):
                self.best_value = current_value
                if self.verbose:
                    print(f"✓ Saved new best model: {self.monitor} = {current_value:.4f}")
            elif self.verbose:
                print(f"✓ Saved checkpoint: {checkpoint_path.name}")
            
            # Update history
            self.history.append({
                'epoch': epoch,
                'metrics': metrics,
                'filename': str(checkpoint_path)
            })
            
            # Save history
            self.save_history()
            
            return True
        
        return False
    
    def save_history(self):
        """Save checkpoint history as JSON"""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def load_checkpoint(self, model, optimizer=None, filename='best_model.pth'):
        """
        Load model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
            filename: Checkpoint filename
        
        Returns:
            Dictionary with epoch and metrics
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.verbose:
            print(f"✓ Loaded checkpoint: {filename}")
            print(f"  Epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"  Metrics: {checkpoint['metrics']}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }

class EarlyStopping:
    """
    Early stopping to stop training when metric stops improving
    """
    
    def __init__(self, patience=10, mode='max', monitor='val_f1', 
                 min_delta=0, verbose=True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait
            mode: 'max' or 'min'
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.mode = mode
        self.monitor = monitor
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False
    
    def is_better(self, current_value):
        """Check if current value is better than best"""
        if self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            return current_value < self.best_value - self.min_delta
    
    def step(self, metrics):
        """
        Check if should stop training
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            True if should stop, False otherwise
        """
        current_value = metrics.get(self.monitor, None)
        
        if current_value is None:
            if self.verbose:
                print(f"⚠️  Warning: {self.monitor} not found in metrics")
            return False
        
        if self.is_better(current_value):
            self.best_value = current_value
            self.counter = 0
            if self.verbose:
                print(f"✓ {self.monitor} improved to {current_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  {self.monitor} did not improve ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered! Best {self.monitor}: {self.best_value:.4f}")
        
        return self.should_stop

if __name__ == '__main__':
    # Example usage
    import torch.nn as nn
    import torch.optim as optim
    
    # Create dummy model
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters())
    
    # Initialize checkpoint manager
    checkpoint = ModelCheckpoint(
        checkpoint_dir='results/checkpoints/test',
        mode='max',
        monitor='val_f1',
        save_best_only=True
    )
    
    # Simulate training
    for epoch in range(1, 6):
        metrics = {
            'train_loss': 0.5 / epoch,
            'val_loss': 0.6 / epoch,
            'val_f1': 0.5 + epoch * 0.05
        }
        
        checkpoint.save_checkpoint(model, optimizer, epoch, metrics)
    
    # Load best checkpoint
    info = checkpoint.load_checkpoint(model, optimizer)
    print(f"\nLoaded epoch {info['epoch']} with metrics: {info['metrics']}")
    
    # Example early stopping
    print("\n" + "="*70)
    print("Early Stopping Example")
    print("="*70)
    
    early_stop = EarlyStopping(patience=3, mode='max', monitor='val_f1')
    
    for epoch in range(1, 10):
        # Simulate plateauing performance
        val_f1 = 0.8 if epoch < 5 else 0.79
        metrics = {'val_f1': val_f1}
        
        if early_stop.step(metrics):
            print(f"Training stopped at epoch {epoch}")
            break
