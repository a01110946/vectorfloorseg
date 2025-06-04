# VectorFloorSeg Project Migration Guide

This guide outlines the steps to migrate and set up the VectorFloorSeg project on a new computer.

## Current Project Status (as of 2025-06-04)

*   **Phase:** Phase 1: Environment Setup.
*   **Git Repository:** Initialized, with the original VecFloorSeg codebase integrated directly into the project root.
*   **Documentation:**
    *   `README.md`, `ARCHITECTURE.md`, `project-structure.md` created and up-to-date.
    *   `docs/phase1_environment_setup.md` updated to reflect the integrated codebase and current setup steps.
*   **Virtual Environment (`vectorfloorseg_env`):** Created on the original machine.
*   **Core Dependencies Installed (on original machine via `requirements.txt`):
    *   `pip` upgraded.
    *   `torch`, `torchvision`, `torchaudio` (with CUDA 11.8 compatibility for PyTorch 2.7.1).
*   **Current Blocker (on original machine):** Installation of PyTorch Geometric (PyG) dependencies (specifically `torch-scatter`) is failing due to the absence of "Microsoft Visual C++ 14.0 or greater" build tools. This is required to compile `torch-scatter` from source as pre-compiled wheels for the current Python/PyTorch version (Python 3.13, PyTorch 2.7.1) were not found.
*   **Next Step after Migration:** Install PyTorch Geometric and its dependencies, then proceed with `setup_custom_pyg.py` and `verify_installation.py` as per `phase1_environment_setup.md`.

## Prerequisites for the New Computer

1.  **Git:** Ensure Git is installed ([https://git-scm.com/downloads](https://git-scm.com/downloads)).
2.  **Python:** Python 3.8 or higher. Python 3.10+ is recommended (the project currently uses Python 3.13 features via PyTorch 2.7.1). Download from [https://www.python.org/downloads/](https://www.python.org/downloads/). Ensure Python and Pip are added to your system's PATH during installation.
3.  **CUDA-Compatible GPU:** Required for PyTorch with GPU support. Ensure your GPU drivers are up to date.
4.  **Microsoft Visual C++ Build Tools:** **Crucial for PyTorch Geometric dependencies.**
    *   Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    *   During installation, select the **"Desktop development with C++"** workload.
    *   A system restart might be necessary after installation.
5.  **Terminal:** Git Bash is recommended (as per user preference for Windows).

## Migration and Setup Steps on the New Computer

1.  **Clone the Repository:**
    *   Open your preferred terminal (e.g., Git Bash).
    *   Navigate to the directory where you want to clone the project.
    *   Run: `git clone <your-repository-url>` (Replace `<your-repository-url>` with the URL of your GitHub repository, e.g., `https://github.com/FernandoMaytorena/vectorfloorseg.git`)
    *   Navigate into the cloned project directory: `cd vectorfloorseg`

2.  **Create and Activate Virtual Environment:**
    *   In the project root (`vectorfloorseg`), create the virtual environment:
        ```bash
        python -m venv vectorfloorseg_env
        ```
    *   Activate the virtual environment:
        *   **Using Git Bash (recommended for Windows):**
            ```bash
            source vectorfloorseg_env/Scripts/activate
            ```
        *   **Using PowerShell/CMD (if Git Bash is not used):**
            ```powershell
            .\vectorfloorseg_env\Scripts\activate
            ```
            (Note: You might need to adjust PowerShell execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)

3.  **Install Dependencies from `requirements.txt`:**
    *   Ensure your virtual environment is active.
    *   Run:
        ```bash
        python -m pip install -r requirements.txt
        ```
    *   This will install `pip`, `torch`, `torchvision`, `torchaudio`, and their dependencies as captured from the previous machine.

4.  **Install PyTorch Geometric (PyG) and its Dependencies:**
    *   This is the step where we were previously blocked.
    *   Ensure Microsoft C++ Build Tools are installed correctly on the new machine.
    *   Follow the instructions in `docs/phase1_environment_setup.md` (Section 1.4 and 1.5) to install PyTorch Geometric 2.0.4 and its sparse dependencies (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`).
    *   The command from the guide, adapted for PyTorch 2.7.1 and CUDA 11.8, is:
        ```bash
        python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
        python -m pip install torch-geometric==2.0.4
        ```
    *   If the above still attempts to build from source and fails (even with C++ tools), try installing `torch-scatter` (and other problematic dependencies one-by-one) with the `--no-build-isolation` flag, which might help the build process find the already installed PyTorch:
        ```bash
        python -m pip install torch-scatter --no-build-isolation
        # ... then torch-sparse, etc. if needed
        ```

5.  **Run Custom PyG Setup Script:**
    *   Once PyG and its dependencies are installed, create and run `setup_custom_pyg.py` as detailed in `docs/phase1_environment_setup.md` (Section 1.6).
    *   The script content is provided in the guide. Create it in the project root.
    *   Run:
        ```bash
        python setup_custom_pyg.py
        ```

6.  **Verify Installation:**
    *   Create and run `verify_installation.py` as detailed in `docs/phase1_environment_setup.md` (Section 1.7).
    *   The script content is provided in the guide. Create it in the project root.
    *   Run:
        ```bash
        python verify_installation.py
        ```

7.  **Install Remaining Dependencies:**
    *   The original `VecFloorSeg` project had other dependencies. While `requirements.txt` captures what was installed so far, you might need to install others listed in `docs/phase1_environment_setup.md` (Section 1.5, "Additional Dependencies") if they weren't pulled in as sub-dependencies of PyTorch/PyG.
    *   Example:
        ```bash
        python -m pip install opencv-python matplotlib scipy Pillow tqdm tensorboard wandb albumentations svglib cairosvg
        ```

8.  **Proceed to Next Phases:**
    *   Once the environment is fully set up and verified, you can proceed with Phase 2: Project Structure Setup and Pretrained Models, as outlined in the main `README.md` and `ARCHITECTURE.md`.

Good luck with the migration! If you encounter issues, refer to the troubleshooting sections in `docs/phase1_environment_setup.md` or seek further assistance.
