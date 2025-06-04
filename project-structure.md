# VectorFloorSeg Project Structure

This document describes the directory and file structure of the VectorFloorSeg project.
The project directly integrates the codebase from the original [VecFloorSeg repository by DrZiji](https://github.com/DrZiji/VecFloorSeg).

## Root Directory (`C:\Users\FernandoMaytorena\GitHub\vectorfloorseg`)

-   **`.git/`**: Git repository data.
-   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (project-specific).
-   **`vectorfloorseg_env/`**: Python virtual environment (if created in project root, not tracked).
-   **`config/`**: Project-specific configuration files.
-   **`docs/`**: Project documentation, including phase guides.
    -   `phase1_environment_setup.md`: Guide for Phase 1.
    -   `(other phase guides)`
-   **`src/`**: Project-specific source code (e.g., extensions, new utilities beyond the original VecFloorSeg).
-   **`tests/`**: Project-specific test scripts (pytest).

---
**Integrated VecFloorSeg Codebase Files & Folders:**
*(These are now at the root level or in their original structure from VecFloorSeg)*

-   **`assets/`**: Assets from VecFloorSeg.
-   **`configs/`**: Original configuration files from VecFloorSeg (e.g., for GraphGym).
-   **`DATA/`**: Placeholder for data, as per VecFloorSeg.
-   **`graphgym_oldversion204/`**: Custom GraphGym library files from VecFloorSeg.
-   **`run_scripts_slurm/`**: SLURM run scripts from VecFloorSeg.
-   **`torch_geometric_oldversion204/`**: Custom PyTorch Geometric library files from VecFloorSeg.
-   **`main_GraphGym.py`**: Main script from VecFloorSeg.
-   **`requirements.txt`**: Original requirements from VecFloorSeg (we will manage project dependencies separately or merge).
-   *(Other files and folders from VecFloorSeg)*

---
**Project Core Files:**

-   **`ARCHITECTURE.md`**: Overview of the system architecture.
-   **`project-structure.md`**: This file.
-   **`README.md`**: Main project readme file (project-specific).
-   **`setup_custom_pyg.py`**: Script to handle custom PyTorch Geometric modifications (to be created in project root as per Phase 1 guide).
-   **`verify_installation.py`**: Script to verify environment setup (to be created in project root as per Phase 1 guide).
-   **(Project-specific `requirements.txt` will be generated later)**

This structure will evolve as the project progresses through its implementation phases.