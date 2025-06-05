# VectorFloorSeg Project Structure

**Status:** Phase 2: Project Structure Setup and Pretrained Models - Partially Complete (Core structure and scripts set up).

This document describes the directory and file structure of the VectorFloorSeg project.
The project directly integrates the codebase from the original [VecFloorSeg repository by DrZiji](https://github.com/DrZiji/VecFloorSeg).

## Root Directory Structure

- **.git/**: Git repository data.
- **.gitignore**: Specifies intentionally untracked files that Git should ignore.
- **README.md**: Main project README.
- **ARCHITECTURE.md**: Overview of the system architecture and development phases.
- **project-structure.md**: This file, detailing the project's file and directory layout.
- **requirements.txt**: Python package dependencies for the project (finalized Phase 1).
- **setup_custom_pyg.py**: Script from Phase 1 to integrate custom PyG components.
- **verify_installation.py**: Script from Phase 1 to verify environment and dependencies.
- **download_backbone.py**: (New in Phase 2) Script to download pretrained backbone models.
- **setup_project.py**: (New in Phase 2) Script to automate project setup tasks (downloads, configs, etc.).

- **docs/**: Project documentation.
    - migration_guide.md: Guide for migrating and setting up the project.
    - phase1_environment_setup.md: Detailed steps and outcomes for Phase 1.
    - phase2_project_structure_and_models.md: Guide for Phase 2.
    - MODEL.md: (To be created) Documents LLM models and configurations.
    - PROMPTS.md: (To be created) Documents prompt templates and strategies.
    - *(Other phase guides and documentation will be added here)*

- **src/**: (New/Organized in Phase 2) Project-specific source code.
    - **data/**: Modules for data loading, processing, and augmentation.
    - **models/**: Model implementations (e.g., VectorFloorSeg architecture).
    - **utils/**: Utility functions.
        - `config.py`: Manages experiment configurations.
        - `logging_utils.py`: Sets up project-wide logging.
        - `visualization.py`: Handles training progress visualization.
    - **losses/**: Custom loss functions.

- **data/**: (New/Organized in Phase 2) Datasets and related files.
    - **raw/**: Original datasets (e.g., R2V, CubiCasa-5k) - *To be populated manually*.
    - **processed/**: Preprocessed data ready for model consumption.
    - **datasets/**: PyTorch Dataset classes.

- **models/**: (New/Organized in Phase 2) Model-related files.
    - **pretrained/**: Pretrained backbone models (e.g., ResNet, VGG).
        - `resnet101-torch.pth`
        - `resnet50-torch.pth`
        - `vgg16-torch.pth`

- **configs/**: (New/Organized in Phase 2) Configuration files for experiments.
    - `baseline.yaml`
    - `lightweight.yaml`
    - `debug.yaml`
    - *(Original VecFloorSeg configs might be here or migrated)*

- **experiments/**: (New in Phase 2) Experiment-specific scripts or configurations.

- **outputs/**: (New in Phase 2) Output files from training and experiments.
    - **checkpoints/**: Saved model checkpoints.
    - **logs/**: Training and experiment logs.
        - `training_log_project_setup_run.log`
    - **visualizations/**: Saved plots and visualizations.

- **notebooks/**: (New in Phase 2) Jupyter notebooks for analysis, exploration, and visualization.

- **tests/**: (New/Organized in Phase 2) Unit and integration tests (using pytest).

- **vectorfloorseg_env/**: Python virtual environment (typically in .gitignore).

---
**Integrated VecFloorSeg Codebase Files & Folders (Original Structure):**
*(These are at the root level or in their original structure from VecFloorSeg, copied directly during initial setup. Some may be progressively refactored or relocated into the new structure above.)*

- **assets/**: Assets from VecFloorSeg (e.g., sample images, fonts).
- **DataPreparation/**: Scripts and utilities for data preparation from VecFloorSeg.
- **graphgym/**: Custom GraphGym library files from VecFloorSeg.
- **Metrics/**: Scripts for performance metrics from VecFloorSeg.
- **project_custom_pyg_source/**: Renamed `torch_geometric` from original repo (Phase 1).
- **Replace_with_CubiCasa/**: Folder related to CubiCasa dataset integration from VecFloorSeg.
- **run_scripts_slurm/**: SLURM run scripts from VecFloorSeg.
- **Utils/**: Original utility scripts from VecFloorSeg. *(To be reviewed and potentially merged/migrated to `src/utils/`)*
- **main_GraphGym.py**: Main script for running GraphGym experiments from VecFloorSeg.
- *(Other miscellaneous files and folders from the original VecFloorSeg repository)*

---

This structure will evolve as the project progresses. The aim of Phase 2 is to establish a clean, modular, and scalable foundation.