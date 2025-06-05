# VectorFloorSeg Implementation - Phase 2: Project Structure Setup

## Overview

This phase focuses on setting up the project structure, downloading required pretrained models, and organizing the codebase for development. Ensure you have completed Phase 1 (Environment Setup) before proceeding.

## Prerequisites

- Completed Phase 1: Environment Setup
- Active virtual environment: `vectorfloorseg_env`
- Located in the `VecFloorSeg` repository directory

## 2.1 Project Directory Structure

Create the recommended directory structure for the VectorFloorSeg project:

```bash
# Create directory structure
mkdir -p src/{data,models,utils,losses}
mkdir -p data/{raw,processed,datasets}
mkdir -p models/pretrained
mkdir -p experiments
mkdir -p outputs/{checkpoints,logs,visualizations}
mkdir -p configs
mkdir -p notebooks
mkdir -p tests
```

The resulting structure should look like:

```
VecFloorSeg/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Model implementations
│   ├── utils/          # Utility functions
│   └── losses/         # Custom loss functions
├── data/
│   ├── raw/            # Original datasets (R2V, CubiCasa-5k)
│   ├── processed/      # Preprocessed data
│   └── datasets/       # Dataset classes
├── models/
│   └── pretrained/     # Pretrained backbone models
├── experiments/        # Experiment configurations
├── outputs/
│   ├── checkpoints/    # Model checkpoints
│   ├── logs/           # Training logs
│   └── visualizations/ # Result visualizations
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks for analysis
└── tests/              # Unit tests
```

## 2.2 Download Pretrained Backbone Models

```python
# File: download_backbone.py
import urllib.request
import os
from pathlib import Path
import torch
import hashlib

def verify_file_hash(filepath: str, expected_hash: str) -> bool:
    """Verify downloaded file integrity using SHA256 hash."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash

def download_resnet_backbone():
    """Download the required ResNet-101 pretrained weights."""
    
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ResNet-101 pretrained on ImageNet
    resnet_url = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
    resnet_path = models_dir / "resnet101-torch.pth"
    resnet_hash = "63fe2227"  # Partial hash from filename
    
    if not resnet_path.exists():
        print("Downloading ResNet-101 pretrained weights...")
        try:
            urllib.request.urlretrieve(resnet_url, resnet_path)
            print(f"✓ Downloaded ResNet-101 to: {resnet_path}")
        except Exception as e:
            print(f"✗ Failed to download ResNet-101: {e}")
            return False
    else:
        print(f"✓ ResNet-101 weights already exist at: {resnet_path}")
    
    # Verify the model can be loaded
    try:
        checkpoint = torch.load(resnet_path, map_location='cpu')
        print(f"✓ ResNet-101 model verified - {len(checkpoint)} parameters")
        return True
    except Exception as e:
        print(f"✗ Failed to load ResNet-101 model: {e}")
        return False

def download_additional_backbones():
    """Download additional backbone options."""
    
    models_dir = Path("models/pretrained")
    
    # ResNet-50 (lighter alternative)
    resnet50_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    resnet50_path = models_dir / "resnet50-torch.pth"
    
    if not resnet50_path.exists():
        print("Downloading ResNet-50 pretrained weights...")
        try:
            urllib.request.urlretrieve(resnet50_url, resnet50_path)
            print(f"✓ Downloaded ResNet-50 to: {resnet50_path}")
        except Exception as e:
            print(f"✗ Failed to download ResNet-50: {e}")
    
    # VGG-16 (original DFPR baseline)
    vgg16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
    vgg16_path = models_dir / "vgg16-torch.pth"
    
    if not vgg16_path.exists():
        print("Downloading VGG-16 pretrained weights...")
        try:
            urllib.request.urlretrieve(vgg16_url, vgg16_path)
            print(f"✓ Downloaded VGG-16 to: {vgg16_path}")
        except Exception as e:
            print(f"✗ Failed to download VGG-16: {e}")

if __name__ == "__main__":
    print("=== Downloading Pretrained Backbone Models ===")
    
    success = download_resnet_backbone()
    if success:
        print("\n=== Downloading Additional Backbones ===")
        download_additional_backbones()
    
    print("\n=== Download Summary ===")
    models_dir = Path("models/pretrained")
    for model_file in models_dir.glob("*.pth"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"✓ {model_file.name}: {size_mb:.1f} MB")
```

## 2.3 Create Configuration Management

```python
# File: src/utils/config.py
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    primal_input_dim: int = 66
    dual_input_dim: int = 66
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    num_room_classes: int = 12
    backbone: str = "resnet101"

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.9
    epochs: int = 200
    boundary_loss_weight: float = 0.5
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"  # cosine, step, plateau

@dataclass
class DataConfig:
    """Data processing configuration."""
    image_size: tuple = (256, 256)
    normalize_coords: bool = True
    extend_lines: bool = True
    use_data_augmentation: bool = True
    flip_probability: float = 0.5
    datasets: list = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["R2V", "CubiCasa-5k"]

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "vectorfloorseg_baseline"
    description: str = "Baseline VectorFloorSeg implementation"
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    device: str = "auto"
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 20
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()

class ConfigManager:
    """Manage experiment configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: ExperimentConfig, filename: str = None) -> Path:
        """Save configuration to YAML file."""
        if filename is None:
            filename = f"{config.name}.yaml"
        
        config_path = self.config_dir / filename
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"✓ Configuration saved to: {config_path}")
        return config_path
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """Load configuration from YAML file."""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Remove nested configs from main dict
        config_dict.pop('model', None)
        config_dict.pop('training', None)
        config_dict.pop('data', None)
        
        return ExperimentConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            **config_dict
        )
    
    def create_default_configs(self):
        """Create default configuration files."""
        
        # Baseline configuration
        baseline_config = ExperimentConfig(
            name="baseline_resnet101",
            description="Baseline VectorFloorSeg with ResNet-101 backbone"
        )
        self.save_config(baseline_config, "baseline.yaml")
        
        # Lightweight configuration
        lightweight_config = ExperimentConfig(
            name="lightweight_resnet50",
            description="Lightweight VectorFloorSeg with ResNet-50 backbone",
            model=ModelConfig(
                hidden_dim=128,
                num_layers=4,
                num_heads=4,
                backbone="resnet50"
            ),
            training=TrainingConfig(
                batch_size=16,
                learning_rate=0.02
            )
        )
        self.save_config(lightweight_config, "lightweight.yaml")
        
        # Debug configuration (small, fast training)
        debug_config = ExperimentConfig(
            name="debug_small",
            description="Debug configuration for quick testing",
            model=ModelConfig(
                hidden_dim=64,
                num_layers=2,
                num_heads=2,
                backbone="resnet50"
            ),
            training=TrainingConfig(
                batch_size=4,
                epochs=10,
                learning_rate=0.001
            )
        )
        self.save_config(debug_config, "debug.yaml")
        
        print("✓ Default configurations created")

# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    
    # Test loading
    config = config_manager.load_config("baseline.yaml")
    print(f"Loaded config: {config.name}")
    print(f"Model hidden dim: {config.model.hidden_dim}")
    print(f"Training batch size: {config.training.batch_size}")
```

## 2.4 Create Project Utilities

```python
# File: src/utils/logging_utils.py
import logging
import sys
from pathlib import Path
from datetime import datetime
import torch
import json

def setup_logging(log_dir: str = "outputs/logs", experiment_name: str = "experiment") -> logging.Logger:
    """Setup logging for training runs."""
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('VectorFloorSeg')
    logger.info(f"Logging initialized - Log file: {log_file}")
    
    return logger

def log_model_info(model: torch.nn.Module, logger: logging.Logger):
    """Log model architecture information."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Architecture:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")

def log_experiment_config(config, logger: logging.Logger):
    """Log experiment configuration."""
    
    logger.info("Experiment Configuration:")
    logger.info(f"  Name: {config.name}")
    logger.info(f"  Description: {config.description}")
    logger.info(f"  Model: {config.model.backbone}, hidden_dim={config.model.hidden_dim}")
    logger.info(f"  Training: lr={config.training.learning_rate}, batch_size={config.training.batch_size}")
    logger.info(f"  Device: {config.device}")
```

```python
# File: src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path

class FloorplanVisualizer:
    """Visualize floorplans and segmentation results."""
    
    def __init__(self, save_dir: str = "outputs/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for room types
        self.room_colors = {
            0: '#FFFFFF',  # Background
            1: '#FF6B6B',  # Bedroom
            2: '#4ECDC4',  # Kitchen
            3: '#45B7D1',  # Bathroom
            4: '#96CEB4',  # Living Room
            5: '#FFEAA7',  # Dining Room
            6: '#DDA0DD',  # Closet
            7: '#98D8C8',  # Balcony
            8: '#F7DC6F',  # Hall/Corridor
            9: '#BB8FCE',  # Other Room
            10: '#85C1E9', # Washing Room
            11: '#F8C471'  # Additional category
        }
    
    def plot_primal_graph(
        self, 
        vertices: torch.Tensor, 
        edges: torch.Tensor, 
        edge_predictions: Optional[torch.Tensor] = None,
        title: str = "Primal Graph",
        save_name: Optional[str] = None
    ):
        """Visualize primal graph (line segments)."""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Convert to numpy
        vertices_np = vertices.cpu().numpy()
        edges_np = edges.cpu().numpy()
        
        # Plot edges
        for i, (start_idx, end_idx) in enumerate(edges_np.T):
            start_pos = vertices_np[start_idx]
            end_pos = vertices_np[end_idx]
            
            # Color based on boundary prediction
            if edge_predictions is not None:
                color = 'red' if edge_predictions[i] > 0.5 else 'blue'
                alpha = 0.8
            else:
                color = 'black'
                alpha = 0.6
            
            ax.plot([start_pos[0], end_pos[0]], 
                   [start_pos[1], end_pos[1]], 
                   color=color, alpha=alpha, linewidth=1)
        
        # Plot vertices
        ax.scatter(vertices_np[:, 0], vertices_np[:, 1], 
                  c='black', s=20, alpha=0.7, zorder=5)
        
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_dual_graph(
        self, 
        regions: List[List[Tuple]], 
        room_predictions: Optional[torch.Tensor] = None,
        room_labels: Optional[torch.Tensor] = None,
        title: str = "Dual Graph",
        save_name: Optional[str] = None
    ):
        """Visualize dual graph (regions)."""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        for i, region in enumerate(regions):
            if len(region) < 3:
                continue
            
            # Determine color
            if room_predictions is not None:
                pred_class = torch.argmax(room_predictions[i]).item()
                color = self.room_colors.get(pred_class, '#CCCCCC')
            elif room_labels is not None:
                true_class = room_labels[i].item()
                color = self.room_colors.get(true_class, '#CCCCCC')
            else:
                color = '#CCCCCC'
            
            # Plot filled polygon
            region_array = np.array(region + [region[0]])  # Close the polygon
            ax.fill(region_array[:, 0], region_array[:, 1], 
                   color=color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add region index
            centroid_x = np.mean([p[0] for p in region])
            centroid_y = np.mean([p[1] for p in region])
            ax.text(centroid_x, centroid_y, str(i), 
                   ha='center', va='center', fontsize=8, weight='bold')
        
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_training_curves(
        self, 
        train_losses: List[float], 
        val_losses: List[float],
        save_name: str = "training_curves"
    ):
        """Plot training and validation loss curves."""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            ax.plot(val_epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        return fig, ax
```

## 2.5 Setup Script

```python
# File: setup_project.py
"""Complete project setup script for VectorFloorSeg."""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from download_backbone import download_resnet_backbone, download_additional_backbones
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logging

def main():
    """Run complete project setup."""
    
    print("=== VectorFloorSeg Project Setup ===")
    
    # 1. Download pretrained models
    print("\n1. Downloading pretrained models...")
    success = download_resnet_backbone()
    if success:
        download_additional_backbones()
    
    # 2. Create default configurations
    print("\n2. Creating default configurations...")
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    
    # 3. Setup logging
    print("\n3. Setting up logging...")
    logger = setup_logging(experiment_name="setup")
    logger.info("Project setup completed successfully")
    
    # 4. Verify project structure
    print("\n4. Verifying project structure...")
    required_dirs = [
        "src/data", "src/models", "src/utils", "src/losses",
        "data/raw", "data/processed", "data/datasets",
        "models/pretrained", "experiments",
        "outputs/checkpoints", "outputs/logs", "outputs/visualizations",
        "configs", "notebooks", "tests"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if not full_path.exists():
            missing_dirs.append(dir_path)
            full_path.mkdir(parents=True, exist_ok=True)
    
    if missing_dirs:
        print(f"Created missing directories: {missing_dirs}")
    
    print("\n✓ Project setup complete!")
    print("\nNext steps:")
    print("- Proceed to Phase 3: Data Preparation Pipeline")
    print("- Download datasets (R2V, CubiCasa-5k) to data/raw/")
    print("- Review configurations in configs/")

if __name__ == "__main__":
    main()
```

## Running Phase 2

Execute the following commands to complete Phase 2:

```bash
# Ensure you're in the VecFloorSeg directory with activated environment
cd VecFloorSeg
source vectorfloorseg_env/bin/activate  # Linux/Mac
# vectorfloorseg_env\Scripts\activate  # Windows

# Create directory structure and download models
python download_backbone.py

# Setup configurations and utilities
python setup_project.py

# Verify setup
ls -la configs/
ls -la models/pretrained/
ls -la outputs/
```

## Expected Outputs

After completing Phase 2, you should have:

```
✓ Downloaded pretrained models:
  - models/pretrained/resnet101-torch.pth
  - models/pretrained/resnet50-torch.pth
  - models/pretrained/vgg16-torch.pth

✓ Created configuration files:
  - configs/baseline.yaml
  - configs/lightweight.yaml
  - configs/debug.yaml

✓ Project structure ready for development

✓ Logging and visualization utilities available
```

## Troubleshooting

### Download Issues
- Check internet connection
- Verify write permissions to `models/pretrained/`
- Try downloading models manually if automatic download fails

### Configuration Issues
- Ensure PyYAML is installed: `pip install pyyaml`
- Check file permissions in `configs/` directory

### Path Issues
- Verify you're running scripts from the project root directory
- Check that all required directories were created

## Completion Status (as of 2025-06-04 23:04)

**All steps outlined in this Phase 2 document have been successfully executed and verified:**

*   **Directory Structure:** All specified directories (`src/`, `data/`, `models/`, `experiments/`, `outputs/`, `configs/`, `notebooks/`, `tests/` and their subdirectories) have been created.
*   **Python Scripts Created:**
    *   `download_backbone.py` (at project root)
    *   `src/utils/config.py`
    *   `src/utils/logging_utils.py`
    *   `src/utils/visualization.py`
    *   `setup_project.py` (at project root)
*   **Pretrained Models Downloaded:** The `download_backbone.py` script (and subsequently `setup_project.py`) successfully downloaded and verified the following models into `models/pretrained/`:
    *   `resnet101-torch.pth`
    *   `resnet50-torch.pth`
    *   `vgg16-torch.pth`
*   **Default Configurations Created:** The `setup_project.py` script successfully generated the following default configuration files in the `configs/` directory:
    *   `baseline.yaml`
    *   `debug.yaml`
    *   `lightweight.yaml`
*   **Logging Initialized:** The `setup_project.py` script initialized logging, creating `outputs/logs/training_log_project_setup_run.log`.
*   **Verification:** The presence and correctness of the downloaded models, configuration files, and log file have been confirmed.
*   **Git Commit:** All changes related to this phase have been committed to the local repository.

The project is now correctly set up as per Phase 2 requirements and is ready for dataset integration and the commencement of Phase 3.

## Next Steps

Proceed to **Phase 3: Data Preparation Pipeline** to implement SVG processing and graph construction.
