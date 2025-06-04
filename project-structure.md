# VectorFloorSeg Project Structure

This document describes the directory and file structure of the VectorFloorSeg project.

## Root Directory

-   `.git/`: Git repository data.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `.venv/`: Python virtual environment (if created in project root).
-   `config/`: Configuration files for the project.
-   `docs/`: Project documentation, including phase guides.
    -   `phase1_environment_setup.md`: Guide for Phase 1.
    -   `(other phase guides to be added)`
-   `src/`: Source code for the VectorFloorSeg system.
    -   `data_preparation/`: Modules for data loading and preprocessing.
    -   `model/`: Implementation of the Two-Stream Graph Attention Network.
    -   `training/`: Scripts for training the model.
    -   `evaluation/`: Scripts for evaluating model performance.
    -   `utils/`: Utility functions.
-   `tests/`: Unit tests and integration tests for the project.
    -   `test_data_preparation/`
    -   `test_model/`
    -   `test_utils/`
-   `ARCHITECTURE.md`: Overview of the system architecture.
-   `project-structure.md`: This file.
-   `README.md`: Main project readme file.
-   `requirements.txt`: Python package dependencies. (To be created)
-   `setup_custom_pyg.py`: Script to handle custom PyTorch Geometric modifications. (To be created as per Phase 1 guide)
-   `verify_installation.py`: Script to verify environment setup. (To be created as per Phase 1 guide)

This structure will evolve as the project progresses through its implementation phases.