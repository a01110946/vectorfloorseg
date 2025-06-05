# File: src/utils/config.py
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from typing import Optional, List, Tuple

@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    backbone: str = "resnet101"
    num_classes: int = 2  # E.g., boundary, non-boundary
    hidden_dim: int = 256
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    nheads: int = 8
    dropout: float = 0.1
    # Add other model-specific parameters as needed

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    batch_size: int = 8
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    momentum: float = 0.9 # if using SGD
    epochs: int = 100
    boundary_loss_weight: float = 0.5
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"  # Options: "cosine", "step", "plateau"
    # Add other training-specific parameters

@dataclass
class DataConfig:
    """Data processing and augmentation configuration."""
    image_size: Tuple[int, int] = (256, 256)
    normalize_coords: bool = True
    extend_lines: bool = True # For SVG processing
    use_data_augmentation: bool = True
    flip_probability: float = 0.5
    # Add other data-specific parameters
    datasets: Optional[List[str]] = field(default_factory=lambda: ["R2V", "CubiCasa5k"])


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    name: str = "default_experiment"
    description: str = "Default experiment configuration"
    device: str = "cuda" # "cuda" or "cpu"
    seed: int = 42
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()

class ConfigManager:
    """Manage experiment configurations using YAML files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    
    def save_config(self, config: ExperimentConfig, filename: str) -> Path:
        """Save configuration to YAML file."""
        config_path = self.config_dir / filename
        
        # Convert dataclass to dict for YAML serialization
        config_dict = {
            "name": config.name,
            "description": config.description,
            "device": config.device,
            "seed": config.seed,
            "model": config.model.__dict__,
            "training": config.training.__dict__,
            "data": config.data.__dict__
        }
        
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
            
        # Reconstruct dataclasses
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Remove nested dataclass dicts before passing to ExperimentConfig
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
        """Create and save default configuration files."""
        
        # Baseline configuration
        baseline_config = ExperimentConfig(
            name="baseline_resnet101",
            description="Baseline VectorFloorSeg with ResNet-101 backbone",
            model=ModelConfig(backbone="resnet101"),
            training=TrainingConfig(batch_size=8, epochs=50)
        )
        self.save_config(baseline_config, "baseline.yaml")
        
        # Lightweight configuration (e.g., for faster testing)
        lightweight_config = ExperimentConfig(
            name="lightweight_resnet50",
            description="Lightweight VectorFloorSeg with ResNet-50 for faster iteration",
            model=ModelConfig(backbone="resnet50", num_encoder_layers=4, num_decoder_layers=4),
            training=TrainingConfig(batch_size=16, epochs=20, learning_rate=0.0005),
            data=DataConfig(image_size=(128,128))
        )
        self.save_config(lightweight_config, "lightweight.yaml")

        # Debug configuration (minimal settings for quick checks)
        debug_config = ExperimentConfig(
            name="debug_config",
            description="Minimal configuration for debugging purposes",
            device="cpu",
            model=ModelConfig(
                backbone="resnet18", # Assuming resnet18 is an option or use a very small custom model
                hidden_dim=64,
                num_encoder_layers=1,
                num_decoder_layers=1,
                nheads=2
            ),
            training=TrainingConfig(
                batch_size=2,
                epochs=1,
                learning_rate=0.001
            ),
            data=DataConfig(image_size=(64,64), use_data_augmentation=False)
        )
        self.save_config(debug_config, "debug.yaml")
        
        print("✓ Default configurations created in 'configs/' directory.")

# Example usage (can be removed or kept for testing)
if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    
    # Test loading
    try:
        loaded_baseline = config_manager.load_config("baseline.yaml")
        print(f"\nSuccessfully loaded 'baseline.yaml':")
        print(f"  Name: {loaded_baseline.name}")
        print(f"  Backbone: {loaded_baseline.model.backbone}")
        print(f"  Batch Size: {loaded_baseline.training.batch_size}")

        loaded_debug = config_manager.load_config("debug.yaml")
        print(f"\nSuccessfully loaded 'debug.yaml':")
        print(f"  Name: {loaded_debug.name}")
        print(f"  Device: {loaded_debug.device}")
        print(f"  Image Size: {loaded_debug.data.image_size}")

    except FileNotFoundError as e:
        print(f"Error loading config: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
