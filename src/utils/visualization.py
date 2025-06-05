# File: src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from pathlib import Path

class TrainingVisualizer:
    """Handles visualization of training progress."""

    def __init__(self, experiment_name: str, save_dir: str = "outputs/visualizations"):
        """Initialize the visualizer.

        Args:
            experiment_name (str): Name of the experiment, used for saving files.
            save_dir (str): Directory to save visualization plots.
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            # Add other metrics as needed, e.g., "train_accuracy", "val_accuracy"
        }
        self.epochs: List[int] = []

    def record_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: Optional[float] = None, 
        **kwargs: float
    ) -> None:
        """Record metrics for a completed epoch.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            val_loss (Optional[float]): Validation loss for the epoch.
            **kwargs: Additional metrics to record (e.g., train_accuracy=0.95).
        """
        self.epochs.append(epoch)
        self.history["train_loss"].append(train_loss)
        if val_loss is not None:
            if "val_loss" not in self.history:
                 self.history["val_loss"] = [] # Ensure list exists
            self.history["val_loss"].append(val_loss)
        
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def plot_losses(
        self, 
        save_name: str = "training_loss_plot", 
        show_plot: bool = False
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Plot training and validation losses.

        Args:
            save_name (str): Filename for the saved plot (without extension).
            show_plot (bool): Whether to display the plot interactively.

        Returns:
            Tuple[Optional[plt.Figure], Optional[plt.Axes]]: Matplotlib figure and axes objects, or (None, None) if no data.
        """
        if not self.epochs or not self.history["train_loss"]:
            print("No data available to plot losses.")
            return None, None

        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.epochs, self.history["train_loss"], label='Training Loss', color='blue', linewidth=2)
        
        if self.history.get("val_loss") and len(self.history["val_loss"]) == len(self.epochs):
            ax.plot(self.epochs, self.history["val_loss"], label='Validation Loss', color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.experiment_name} - Training Progress')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150)
        print(f"✓ Loss plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig) # Close the figure to free memory if not showing
            
        return fig, ax

    # Add methods for other plots (e.g., accuracy, learning rate schedule) as needed.

if __name__ == '__main__':
    # Example Usage
    visualizer = TrainingVisualizer(experiment_name="test_viz_experiment")

    # Simulate some training epochs
    for epoch in range(1, 11):
        train_loss = 1.0 / epoch + np.random.rand() * 0.1
        val_loss = 1.0 / epoch + 0.05 + np.random.rand() * 0.05
        visualizer.record_epoch(epoch, train_loss, val_loss, train_accuracy=0.8 + epoch*0.01, val_accuracy=0.78 + epoch*0.01)

    visualizer.plot_losses(show_plot=False)

    # Example with only training loss
    visualizer_train_only = TrainingVisualizer(experiment_name="test_train_only")
    for epoch in range(1,6):
        train_loss = 0.5 / epoch
        visualizer_train_only.record_epoch(epoch, train_loss)
    visualizer_train_only.plot_losses()
    print("Visualization example complete.")
