# VectorFloorSeg Project Structure

This document describes the directory and file structure of the VectorFloorSeg project.
The project directly integrates the codebase from the original [VecFloorSeg repository by DrZiji](https://github.com/DrZiji/VecFloorSeg).

## Root Directory (`C:\Users\FernandoMaytorena\GitHub\vectorfloorseg`)

-   **`.git/`**: Git repository data.
-   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (project-specific).
-   **`vectorfloorseg_env/`**: Python virtual environment (if created in project root, not tracked by Git).
-   **`config/`**: Project-specific configuration files (e.g., for custom workflows, not original VecFloorSeg configs).
-   **`docs/`**: Project documentation.
    -   `phase1_environment_setup.md`: Guide for Phase 1.
    -   `migration_guide.md`: Guide for migrating the project to a new computer.
    -   *(Other phase guides and documentation will be added here)*
-   **`src/`**: Project-specific source code (e.g., extensions, new utilities beyond the original VecFloorSeg). Empty for now.
-   **`tests/`**: Project-specific test scripts (pytest). Empty for now.

---
**Integrated VecFloorSeg Codebase Files & Folders:**
*(These are at the root level or in their original structure from VecFloorSeg, copied directly)*

-   **`assets/`**: Assets from VecFloorSeg (e.g., sample images, fonts).
-   **`configs/`**: Original configuration files from VecFloorSeg (e.g., for GraphGym experiments).
-   **`DataPreparation/`**: Scripts and utilities for data preparation from VecFloorSeg.
-   **`graphgym/`**: Custom GraphGym library files from VecFloorSeg (Note: directory name confirmed as `graphgym`).
-   **`Metrics/`**: Scripts for performance metrics from VecFloorSeg.
-   **`Replace_with_CubiCasa/`**: Folder related to CubiCasa dataset integration from VecFloorSeg.
-   **`run_scripts_slurm/`**: SLURM run scripts from VecFloorSeg (may not be used directly).
-   **`torch_geometric/`**: Custom PyTorch Geometric library files from VecFloorSeg (Note: directory name confirmed as `torch_geometric`).
-   **`Utils/`**: Utility scripts from VecFloorSeg.
-   **`main_GraphGym.py`**: Main script for running GraphGym experiments from VecFloorSeg.
-   *(Other miscellaneous files and folders from the original VecFloorSeg repository)*

---
**Project Core Files (at root):**

-   **`ARCHITECTURE.md`**: Overview of the system architecture and development phases.
-   **`project-structure.md`**: This file, detailing the project's file and directory layout.
-   **`README.md`**: Main project readme file, providing an overview and setup instructions.
-   **`requirements.txt`**: Project-specific Python dependencies file, generated to capture the current state of `vectorfloorseg_env`. This should be used for setting up the environment on new machines.
-   **`setup_custom_pyg.py`**: Script to handle custom PyTorch Geometric modifications (to be created in project root as per Phase 1 guide).
-   **`verify_installation.py`**: Script to verify environment setup (to be created in project root as per Phase 1 guide).

This structure will evolve as the project progresses through its implementation phases.
