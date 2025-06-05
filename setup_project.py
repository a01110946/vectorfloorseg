# File: setup_project.py
"""Complete project setup script for VectorFloorSeg.

This script automates:
1. Downloading pretrained backbone models.
2. Creating default experiment configuration files.
3. Setting up project-wide logging.
4. Verifying and creating the necessary directory structure.
"""

import sys
from pathlib import Path

# Add src to Python path for local imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

# Imports from our project modules (must be after sys.path modification)
# Assuming download_backbone.py is at the project root alongside this script
import download_backbone 
from utils.config import ConfigManager
from utils.logging_utils import setup_logging

def main() -> None:
    """Run the complete project setup procedures."""
    
    print("=== VectorFloorSeg Project Setup Starting ===")
    
    # 1. Download pretrained models
    print("\n--- 1. Downloading Pretrained Models ---")
    # Call functions from download_backbone.py
    # download_backbone.py itself has an if __name__ == "__main__": block,
    # but here we call its functions directly.
    resnet_success = download_backbone.download_resnet_backbone()
    if resnet_success:
        download_backbone.download_additional_backbones()
        print("✓ Pretrained model download process completed.")
    else:
        print("✗ ResNet-101 download/verification failed. Check messages above.")
    
    # 2. Create default configurations
    print("\n--- 2. Creating Default Configurations ---")
    # ConfigManager will create files in 'configs/' relative to project_root
    config_manager = ConfigManager(config_dir=str(project_root / "configs"))
    config_manager.create_default_configs()
    print("✓ Default configurations created (baseline.yaml, lightweight.yaml, debug.yaml).")
    
    # 3. Setup logging
    print("\n--- 3. Setting Up Logging ---")
    # setup_logging will create logs in 'outputs/logs/' relative to project_root
    logger = setup_logging(
        experiment_name="project_setup_run", 
        log_dir=str(project_root / "outputs/logs")
    )
    logger.info("Project setup process initiated by setup_project.py.")
    
    # 4. Verify project structure (and create if missing)
    print("\n--- 4. Verifying Project Directory Structure ---")
    required_dirs_relative = [
        "src/data", "src/models", "src/utils", "src/losses",
        "data/raw", "data/processed", "data/datasets",
        "models/pretrained", 
        "experiments",
        "outputs/checkpoints", "outputs/logs", "outputs/visualizations",
        "configs", "notebooks", "tests"
    ]
    
    missing_dirs_created_count = 0
    for dir_path_str_relative in required_dirs_relative:
        full_path = project_root / dir_path_str_relative
        if not full_path.exists():
            logger.info(f"Directory not found, creating: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
            missing_dirs_created_count += 1
            print(f"   Created: {full_path}")
    
    if missing_dirs_created_count > 0:
        logger.info(f"Created {missing_dirs_created_count} missing directories during verification.")
        print(f"✓ {missing_dirs_created_count} missing directories were created.")
    else:
        logger.info("All required directories already exist.")
        print("✓ All required directories confirmed to exist.")
    
    logger.info("Project setup script completed successfully.")
    print("\n=== VectorFloorSeg Project Setup Complete! ===")
    print("\nNext steps (as per Phase 2 document):")
    print("  - Review configurations in the 'configs/' directory.")
    print("  - Check downloaded models in the 'models/pretrained/' directory.")
    print("  - Manually download datasets (e.g., R2V, CubiCasa-5k) to 'data/raw/' as per project requirements.")
    print("  - You can run 'python download_backbone.py' and 'python setup_project.py' manually if needed.")
    print("  - Proceed to Phase 3: Data Preparation Pipeline.")

if __name__ == "__main__":
    main()
