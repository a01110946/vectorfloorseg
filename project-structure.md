# VectorFloorSeg Project Structure

**Status:** Phase 1: Environment Setup - Complete.

This document describes the directory and file structure of the VectorFloorSeg project.
The project directly integrates the codebase from the original [VecFloorSeg repository by DrZiji](https://github.com/DrZiji/VecFloorSeg).

## Root Directory Structure

- **.git/**: Git repository data.
- **.gitignore**: Specifies intentionally untracked files that Git should ignore (e.g., ectorfloorseg_env/, __pycache__/, *.pyc, wandb/).
- **README.md**: Main project README (to be updated with project-specific details).
- **docs/**: Project documentation.
    - migration_guide.md: Guide for migrating and setting up the project.
    - phase1_environment_setup.md: Detailed steps and outcomes for Phase 1.
    - phase2_project_structure_and_models.md: (Placeholder for Phase 2).
    - MODEL.md: (To be created, will document LLM models and configurations).
    - PROMPTS.md: (To be created, will document prompt templates and strategies).
    - *(Other phase guides and documentation will be added here)*
- **src/**: Project-specific source code (e.g., extensions, new utilities beyond the original VecFloorSeg). Currently empty.
- **	ests/**: Project-specific test scripts (using pytest). Currently empty.
- **ectorfloorseg_env/**: Python virtual environment for this project. (Note: This directory is typically included in .gitignore and not tracked in the repository. Its creation and management are detailed in docs/phase1_environment_setup.md).

---
**Integrated VecFloorSeg Codebase Files & Folders:**
*(These are at the root level or in their original structure from VecFloorSeg, copied directly. Some may have been modified or relocated as noted.)*

- **ssets/**: Assets from VecFloorSeg (e.g., sample images, fonts).
- **configs/**: Original configuration files from VecFloorSeg (e.g., for GraphGym experiments).
- **DataPreparation/**: Scripts and utilities for data preparation from VecFloorSeg.
- **graphgym/**: Custom GraphGym library files from VecFloorSeg. Its contents are used by setup_custom_pyg.py.
- **Metrics/**: Scripts for performance metrics from VecFloorSeg.
- **project_custom_pyg_source/**: Contains the custom PyTorch Geometric source code originally from the VecFloorSeg project. This directory was named 	orch_geometric in the original repository and was **renamed during Phase 1** to avoid import conflicts with the installed PyG version. Its contents are used by setup_custom_pyg.py.
- **Replace_with_CubiCasa/**: Folder related to CubiCasa dataset integration from VecFloorSeg.
- **un_scripts_slurm/**: SLURM run scripts from VecFloorSeg (may not be used directly).
- **Utils/**: Utility scripts from VecFloorSeg.
- **main_GraphGym.py**: Main script for running GraphGym experiments from VecFloorSeg.
- *(Other miscellaneous files and folders from the original VecFloorSeg repository)*

---
**Project Core Files (at root):**

- **ARCHITECTURE.md**: Overview of the system architecture and development phases. (To be updated as per project evolution).
- **project-structure.md**: This file, detailing the project's file and directory layout.
- **equirements.txt**: Python package dependencies for the project. This file was generated and finalized at the end of Phase 1 to capture the current state of ectorfloorseg_env.
- **setup_custom_pyg.py**: Python script created and used during Phase 1 to integrate the custom PyTorch Geometric and GraphGym components from project_custom_pyg_source/ and graphgym/ into the installed PyG version.
- **erify_installation.py**: Python script created and used during Phase 1 to verify the correctness of the Python environment and all dependencies.

This structure will evolve as the project progresses through its implementation phases.
