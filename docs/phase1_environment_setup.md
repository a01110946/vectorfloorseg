# VectorFloorSeg Implementation - Phase 1: Environment Setup and Dependencies

## 1. Overview

This document provides a comprehensive guide to setting up the Python development environment for the VectorFloorSeg project on a Windows machine. It details the installation of Python, PyTorch with CUDA support, the specific PyTorch Geometric (PyG) version 2.0.4, and all other necessary project dependencies.

This guide incorporates learnings and troubleshooting steps encountered during the initial setup, aiming to provide a clear path and preempt potential issues for future setups. The primary package manager used is pip within a Python virtual environment.

**Key Goals of this Phase:**
*   Establish a stable and reproducible Python environment.
*   Install PyTorch with GPU (CUDA) acceleration.
*   Correctly install PyTorch Geometric 2.0.4 and its sparse dependencies.
*   Address and document resolutions for common installation challenges, particularly with PyG and CairoSVG.
*   Install all other Python packages required by the project.
*   Verify the complete setup using a custom script.
*   Finalize a 
equirements.txt file reflecting the verified environment.

## 2. Prerequisites

Before starting, ensure your system meets the following requirements:

*   **Operating System:** Windows 10/11.
*   **Python:** Python 3.11.x (this guide used 3.11.9). Ensure Python is added to your system PATH during installation.
*   **Git:** Git for Windows (provides Git Bash, a recommended terminal).
*   **NVIDIA GPU:** An NVIDIA GPU compatible with CUDA 11.8.
*   **NVIDIA CUDA Toolkit & Drivers:**
    *   While PyTorch with CUDA support often bundles necessary CUDA runtime libraries, it's good practice to have up-to-date NVIDIA drivers. The full CUDA Toolkit installation is generally not required if using PyTorch's pre-compiled binaries, but ensure your drivers support CUDA 11.8.
*   **Microsoft C++ Build Tools:** Required for compiling some Python packages from source.
    *   Install via the "Visual Studio Installer". Select "Desktop development with C++" workload.

## 3. Virtual Environment Setup

Using a virtual environment is crucial for isolating project dependencies.

1.  **Navigate to your project root directory** (e.g., C:\Users\YourUser\OneDrive\Documentos\GitHub\vectorfloorseg).
2.  **Create the virtual environment:**
    `bash
    python -m venv vectorfloorseg_env
    `
3.  **Activate the virtual environment:**
    *   In **Git Bash**:
        `bash
        source ./vectorfloorseg_env/Scripts/activate
        `
    *   In **Windows Command Prompt or PowerShell**:
        `bash
        .\vectorfloorseg_env\Scripts\Activate.ps1  # PowerShell (ensure execution policy allows scripts)
        .\vectorfloorseg_env\Scripts\activate.bat   # Command Prompt
        `
    Your terminal prompt should now indicate that the (vectorfloorseg_env) is active.

## 4. PyTorch Installation (Version 2.7.1 with CUDA 11.8)

Install PyTorch, torchvision, and torchaudio, specifying the version and CUDA compatibility.

`bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
`

*   **Verification (Optional but Recommended):**
    Create a temporary Python script (e.g., check_torch.py) or run directly in Python interpreter:
    `python
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    `

## 5. PyTorch Geometric (PyG) 2.0.4 Installation

This is a multi-step process requiring specific versions for compatibility with PyTorch 2.7.1 and CUDA 11.8.

### 5.1. Install Sparse Dependencies

PyG relies on several sparse matrix libraries. These must be installed first, matching your PyTorch and CUDA versions. The -f flag directs pip to find wheels at the specified URL.

`bash
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
pip install torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
pip install torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
pip install torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
`
*   **Note:** The PyTorch version in the URL (e.g., 	orch-2.7.1+cu118) must exactly match your installed PyTorch version for these pre-compiled wheels. If a minor version mismatch occurs (e.g., you have 2.7.1 but wheels are only for 2.7.0), you might need to adjust the URL or PyTorch version slightly, but compatibility is key.

### 5.2. Install PyTorch Geometric Core Package

Once sparse dependencies are in place, install PyG version 2.0.4:

`bash
pip install torch-geometric==2.0.4
`

### 5.3. Critical Learning: Resolving PyG Import Conflict

A significant issue was encountered where Python would import PyG from a local directory within the project root (vectorfloorseg/torch_geometric) instead of the correctly installed package in the virtual environment's site-packages. This local directory, part of the original VecFloorSeg codebase, might be an incompatible or incomplete version (e.g., it reported as 2.0.5 but lacked key submodules like torch_geometric.profile).

*   **Symptoms:** ModuleNotFoundError for submodules like torch_geometric.profile, or torch_geometric.__version__ reporting an unexpected version.
*   **Reasoning:** Python's import system prioritizes directories in the current working path (or sys.path) before site-packages. A local directory named torch_geometric shadows the installed package.
*   **Solution:** Rename the local torch_geometric directory in the project root.
    `bash
    # In the project root directory (e.g., using Git Bash)
    mv torch_geometric project_custom_pyg_source
    `
    This renaming (to project_custom_pyg_source) ensures that Python correctly finds and imports the PyG 2.0.4 installation from the virtual environment.

## 6. Handling Custom PyG / GraphGym Components

The original VecFloorSeg project includes custom modifications to PyTorch Geometric and GraphGym. These are now located in:
*   project_custom_pyg_source/ (formerly torch_geometric/ at project root)
*   graphgym/ (at project root)

A script, setup_custom_pyg.py, was developed to integrate these custom components into the active PyG installation.

*   **Current Approach for Phase 1:** For the initial environment setup and verification, the priority was to establish a working, standard PyG 2.0.4 installation. The setup_custom_pyg.py script was **not run** after renaming project_custom_pyg_source.
*   **Future Use:** If the specific customizations from VecFloorSeg are required for project functionality, setup_custom_pyg.py can be reviewed and run. This typically involves backing up the site-packages/torch_geometric and site-packages/graphgym directories and replacing them with the versions from project_custom_pyg_source/ and graphgym/.

## 7. Installation of Other Project Dependencies

Several other Python packages are required.

1.  **Install main additional packages:**
    `bash
    pip install opencv-python matplotlib tensorboard wandb albumentations svglib CairoSVG
    `
2.  **Ensure base packages are present:** Packages like 
umpy, scipy, Pillow, 	qdm are often installed as dependencies of the above or PyTorch/PyG. They will be captured in the final 
equirements.txt.

### 7.1. Critical Learning: CairoSVG Setup on Windows

CairoSVG is used for SVG file manipulation and has an external C library dependency.

*   **Issue:** After pip install CairoSVG, you may encounter OSError: no library called "cairo-2" was found or similar errors related to libcairo-2.dll not being found.
*   **Reasoning:** The CairoSVG Python package is a wrapper around the Cairo 2D graphics library. This C library must be separately installed on your system and accessible via the system's PATH.
*   **Solution:**
    1.  **Install GTK for Windows:** The Cairo library is commonly distributed with GTK. The recommended way is to use MSYS2:
        *   Install MSYS2 from [https://www.msys2.org/](https://www.msys2.org/).
        *   Open an MSYS2 terminal and run:
            `bash
            pacman -S mingw-w64-x86_64-gtk3
            `
    2.  **Add GTK bin to System PATH:**
        *   Locate the bin directory of your MSYS2 MinGW64 installation (e.g., C:\msys64\mingw64\bin). This directory contains libcairo-2.dll and its dependencies.
        *   Add this full path to your Windows System PATH environment variable.
    3.  **Restart:** Close and reopen all terminals, IDEs, and command prompts. A system restart might be necessary for the PATH changes to be fully effective.

## 8. Verification Script (verify_installation.py)

To ensure all components are correctly installed and accessible, a script verify_installation.py was created in the project root.

*   **Purpose:** Checks Python version, PyTorch version and CUDA availability, PyG version and location, versions of sparse dependencies, and import success for all other key packages.
*   **Usage:**
    `bash
    # Ensure your virtual environment is active
    python verify_installation.py
    `
*   **Expected Outcome:** The script should report success () for all checks. This confirms the environment is correctly configured.

## 9. Finalizing requirements.txt

Once all dependencies are installed and the verify_installation.py script confirms a successful setup, generate/update the requirements.txt file to capture the state of the virtual environment.

`bash
# Ensure your virtual environment is active
pip freeze > requirements.txt
`
This requirements.txt file should be committed to Git and used for replicating the environment in the future.

## 10. Troubleshooting Summary & Key Learnings

*   **PyG Import Conflict:** A local torch_geometric directory can shadow the installed package. **Solution:** Rename the local directory (e.g., to project_custom_pyg_source).
*   **CairoSVG Missing DLLs on Windows:** CairoSVG needs the Cairo C library. **Solution:** Install GTK for Windows (via MSYS2 is recommended) and add its bin directory to the system PATH.
*   **WinError 32 during pip install:** This indicates a file lock, often because a Python process or IDE is using files in the virtual environment. **Solution:** Close any running Python scripts, Jupyter notebooks, IDEs (like VS Code if it's accessing the venv), or terminal sessions that might be holding locks, then retry the pip install command.
*   **PyG Sparse Dependency Versions:** Always use versions compatible with your specific PyTorch and CUDA setup. The -f <URL> flag with pip install is crucial for pointing to the correct pre-compiled wheels.
*   **Virtual Environment Integrity:** Always ensure your virtual environment is activated before running pip commands or Python scripts.

This completes Phase 1: Environment Setup. The system should now be ready for subsequent development phases.
