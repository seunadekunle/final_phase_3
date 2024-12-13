"""Early stopping utility to prevent overfitting."""

class EarlyStopping:
    """Early stopping handler class.
    
    Monitors validation loss and stops training when no improvement is seen
    for a specified number of epochs.
    """
    
    def __init__(self, patience, min_delta):
        """Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_acc = 0.0
    
    def __call__(self, val_loss):
        """Check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return False 