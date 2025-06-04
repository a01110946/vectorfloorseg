# VectorFloorSeg Implementation - Phase 1: Environment Setup and Dependencies

## Overview

This document covers the complete environment setup for VectorFloorSeg using pip and virtual environments. The codebase from the original `DrZiji/VecFloorSeg` repository has been directly integrated into this project's root directory. The main challenge is handling the custom PyTorch Geometric modifications that the project requires.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git for cloning repositories

## 1.1 Create Virtual Environment

```bash
# Ensure you are in the project root directory: C:\Users\FernandoMaytorena\GitHub\vectorfloorseg
# Create a new virtual environment
python -m venv vectorfloorseg_env

# Activate the environment
# On Linux/Mac:
# source vectorfloorseg_env/bin/activate
# On Windows (Git Bash):
source vectorfloorseg_env/Scripts/activate
# On Windows (Command Prompt/PowerShell):
# vectorfloorseg_env\Scripts\activate
```

## 1.2 Integrate VecFloorSeg Code and Install Base Dependencies

The codebase from the original `DrZiji/VecFloorSeg` repository has been directly integrated into this project's root directory. The following steps should be performed from the project root (`C:\Users\FernandoMaytorena\GitHub\vectorfloorseg`).

```bash
# Ensure you are in the project root directory
# cd C:\Users\FernandoMaytorena\GitHub\vectorfloorseg

# (Activate your virtual environment 'vectorfloorseg_env' if not already active)
# source vectorfloorseg_env/Scripts/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install PyTorch first (adjust CUDA version as needed for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 1.3 Install PyTorch Geometric 2.0.4

The project specifically requires PyG 2.0.4, which will then be customized.

```bash
# Install PyTorch Geometric 2.0.4 and related packages
# (Ensure your virtual environment 'vectorfloorseg_env' is activated)
pip install torch-geometric==2.0.4
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu118.html
```

## 1.4 Handle Custom PyG Modifications

The critical step is replacing standard PyG components with custom ones provided in the integrated `VecFloorSeg` codebase. The `setup_custom_pyg.py` script (to be created in the project root) will handle this.

```python
# File: setup_custom_pyg.py (to be created in project root)
import os
import shutil
import site
import sys
from pathlib import Path

def find_pyg_installation():
    """Find the PyTorch Geometric installation directory in site-packages."""
    # Check standard site-packages locations
    site_packages_dirs = site.getsitepackages()
    
    # Also check user site-packages, as pip might install there
    user_site_packages = site.getusersitepackages()
    if user_site_packages not in site_packages_dirs:
        site_packages_dirs.append(user_site_packages)

    for sp_dir in site_packages_dirs:
        pyg_path = Path(sp_dir) / "torch_geometric"
        if pyg_path.exists() and pyg_path.is_dir():
            return pyg_path
            
    # Fallback for some environments like virtualenvs if not found above
    # sys.prefix points to the root of the virtual environment
    venv_sp_path = Path(sys.prefix) / "lib" / f"python{sys.version_major}.{sys.version_minor}" / "site-packages" / "torch_geometric"
    if venv_sp_path.exists() and venv_sp_path.is_dir():
        return venv_sp_path
        
    return None

def backup_and_replace(src_dir: Path, dest_dir: Path, backup_base_dir: Path):
    """Backs up original components from dest_dir and replaces them with components from src_dir."""
    if not dest_dir.exists():
        print(f"Error: Destination directory {dest_dir} does not exist. PyG installation might be incomplete.")
        return

    for item in src_dir.iterdir():
        dest_item_path = dest_dir / item.name
        backup_item_path = backup_base_dir / dest_dir.name / item.name

        # Ensure backup subdirectory exists
        backup_item_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_item_path.exists():
            # Backup existing item (file or directory)
            if dest_item_path.is_dir():
                if not backup_item_path.exists(): # Avoid re-backing up if script is run multiple times
                    shutil.copytree(dest_item_path, backup_item_path, dirs_exist_ok=True)
                shutil.rmtree(dest_item_path) # Remove original directory
            else: # It's a file
                if not backup_item_path.exists():
                    shutil.copy2(dest_item_path, backup_item_path) # copy2 preserves metadata
                os.remove(dest_item_path) # Remove original file
            print(f"  Backed up '{dest_item_path.name}' to '{backup_item_path}'")

        # Copy new item
        if item.is_dir():
            shutil.copytree(item, dest_item_path, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item_path)
        print(f"  Replaced '{dest_item_path.name}' with custom version from '{item}'")

def main():
    pyg_install_path = find_pyg_installation()
    if not pyg_install_path:
        print("Error: PyTorch Geometric installation not found. Please ensure PyG 2.0.4 is installed.")
        sys.exit(1)
    
    print(f"Found PyTorch Geometric installation at: {pyg_install_path}")

    # Define source paths for custom components (relative to this script, now at project root)
    # These are the folders containing the modified PyG/GraphGym code,
    # integrated from the original VecFloorSeg repository.
    project_root = Path(__file__).parent.resolve()
    CUSTOM_PYG_SOURCE_DIR = project_root / "torch_geometric" # Custom PyG components
    CUSTOM_GRAPHGYM_SOURCE_DIR = project_root / "graphgym"   # Custom GraphGym components

    # Backup directory (created in site-packages alongside PyG for clarity)
    pyg_parent_dir = pyg_install_path.parent
    backup_dir_base = pyg_parent_dir / "torch_geometric_backup_original"
    
    print(f"Original components will be backed up under: {backup_dir_base}")

    # 1. Handle PyTorch Geometric custom components
    if CUSTOM_PYG_SOURCE_DIR.exists() and CUSTOM_PYG_SOURCE_DIR.is_dir():
        print(f"\nProcessing custom PyTorch Geometric components from: {CUSTOM_PYG_SOURCE_DIR}")
        # We are replacing the entire torch_geometric directory in site-packages
        # with the custom one from our project.
        
        # First, backup the original torch_geometric directory from site-packages
        original_pyg_backup_path = backup_dir_base / "torch_geometric_original_full"
        if pyg_install_path.exists() and not original_pyg_backup_path.exists():
            original_pyg_backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(pyg_install_path, original_pyg_backup_path, dirs_exist_ok=True)
            print(f"  Backed up original 'torch_geometric' directory to '{original_pyg_backup_path}'")
        
        # Remove the original torch_geometric directory from site-packages
        shutil.rmtree(pyg_install_path)
        print(f"  Removed original 'torch_geometric' directory from '{pyg_install_path}'")
        
        # Copy the custom torch_geometric directory to site-packages
        shutil.copytree(CUSTOM_PYG_SOURCE_DIR, pyg_install_path, dirs_exist_ok=True)
        print(f"  Copied custom 'torch_geometric' directory to '{pyg_install_path}'")
        
    else:
        print(f"Error: Custom PyG source directory not found: {CUSTOM_PYG_SOURCE_DIR}")
        print("Ensure the 'torch_geometric' directory (from VecFloorSeg) is at the project root.")
        sys.exit(1)

    # 2. Handle GraphGym custom components (if they exist and are needed)
    # GraphGym is often a sub-component or used alongside PyG.
    # The original VecFloorSeg might have custom GraphGym parts.
    # We assume these would also go into the site-packages, potentially under pyg_install_path / 'graphgym'
    # or a separate 'graphgym' in site-packages if it's a standalone install.
    # For this example, let's assume custom graphgym components are to be copied into the *newly placed* custom torch_geometric's graphgym folder.
    
    pyg_graphgym_dest_path = pyg_install_path / "graphgym" # Destination for graphgym components within the custom PyG

    if CUSTOM_GRAPHGYM_SOURCE_DIR.exists() and CUSTOM_GRAPHGYM_SOURCE_DIR.is_dir():
        print(f"\nProcessing custom GraphGym components from: {CUSTOM_GRAPHGYM_SOURCE_DIR}")
        backup_graphgym_dir = backup_dir_base / "graphgym_original_within_pyg"
        # The backup_and_replace function handles sub-components
        backup_and_replace(CUSTOM_GRAPHGYM_SOURCE_DIR, pyg_graphgym_dest_path, backup_graphgym_dir)
    else:
        print(f"\nWarning: Custom GraphGym source directory '{CUSTOM_GRAPHGYM_SOURCE_DIR}' not found.")
        print("Ensure the 'graphgym' directory (from VecFloorSeg) is at the project root if custom GraphGym components are needed.")
        print("Proceeding without custom GraphGym components if not found or not applicable.")

    print("\nCustom PyG and GraphGym setup complete.")
    print(f"Original files are backed up in subdirectories under: {backup_dir_base}")
    print("To restore, copy backed-up 'torch_geometric_original_full' back to site-packages, replacing the custom one.")
    print("And similarly for GraphGym if it was customized.")

if __name__ == "__main__":
    main()
```

## 1.5 Create `requirements.txt` and Install Other Dependencies

The original `VecFloorSeg` repository contains a `requirements.txt`. We will use this as a basis.
First, copy the content of the `requirements.txt` from the integrated `VecFloorSeg` code into a new `requirements.txt` at the project root if it's not already there, or ensure the existing one is from `VecFloorSeg`. Then install these dependencies.

```bash
# (Ensure your virtual environment 'vectorfloorseg_env' is activated)

# If you haven't already, ensure requirements.txt from VecFloorSeg is at the project root.
# Then install additional dependencies:
pip install -r requirements.txt

# The phase1 guide also lists these, ensure they are covered by requirements.txt or install explicitly:
# pip install opencv-python matplotlib numpy scipy Pillow tqdm tensorboard wandb albumentations svglib cairosvg
# (It's better if these are in requirements.txt)
```
*Self-correction: The `requirements.txt` from VecFloorSeg likely already covers these. We will add it to git later.*

## 1.6 Create and Run Custom PyG Setup Script

1.  Create the `setup_custom_pyg.py` file in your project root (`C:\Users\FernandoMaytorena\GitHub\vectorfloorseg`) with the Python code provided in section 1.4.
2.  Run the script:
    ```bash
    # (Ensure your virtual environment 'vectorfloorseg_env' is activated)
    python setup_custom_pyg.py
    ```

## 1.7 Create and Run Verification Script

1.  Create a `verify_installation.py` file in your project root with the following content:
    ```python
    # File: verify_installation.py
    import sys
    import importlib

    print(f"Python version: {sys.version}")

    def check_package(package_name, custom_check=None):
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'N/A')
            print(f"✓ {package_name} (Version: {version}) imported successfully.")
            if custom_check:
                custom_check(module)
        except ImportError:
            print(f"✗ {package_name} not found.")
        except Exception as e:
            print(f"✗ Error importing or checking {package_name}: {e}")

    def check_pytorch_cuda(torch_module):
        cuda_available = torch_module.cuda.is_available()
        print(f"  CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA Version: {torch_module.version.cuda}")
            print(f"  GPU Count: {torch_module.cuda.device_count()}")
            print(f"  Current GPU: {torch_module.cuda.get_device_name(torch_module.cuda.current_device())}")

    def check_pyg_custom_components(pyg_module):
        # Example: Check for a function/class known to be in custom version
        # This is highly dependent on what VecFloorSeg customized.
        # For now, we'll just note its presence.
        # Add specific checks here if you know what to look for.
        print(f"  PyTorch Geometric location: {pyg_module.__path__}")
        # A simple check could be to see if the path is now pointing to our project's dir,
        # but setup_custom_pyg.py copies files *into* site-packages, so path won't change.
        # Instead, one might check for a specific modified file's content or a unique function.
        # For now, we assume if setup_custom_pyg.py ran, it's 'customized'.
        print("  (Assuming custom components are integrated if setup_custom_pyg.py ran successfully)")


    print("\nChecking core dependencies:")
    check_package("torch", custom_check=check_pytorch_cuda)
    check_package("torchvision")
    check_package("torchaudio")
    check_package("torch_geometric", custom_check=check_pyg_custom_components)
    check_package("torch_scatter")
    check_package("torch_sparse")
    check_package("torch_cluster")
    check_package("torch_spline_conv")

    print("\nChecking other dependencies:")
    other_packages = [
        "cv2", "matplotlib", "numpy", "scipy", "PIL", 
        "tqdm", "tensorboard", "wandb", "albumentations", 
        "svglib", "cairosvg"
    ]
    for pkg in other_packages:
        check_package(pkg)
        
    print("\nVerification script finished.")
    print("Review the output above. If all checks (✓) pass and CUDA is available (if intended), setup is likely correct.")
    print("Ensure to also check for specific custom PyG component functionality if known.")
    ```
2.  Run the verification script:
    ```bash
    # (Ensure your virtual environment 'vectorfloorseg_env' is activated)
    python verify_installation.py
    ```

## Summary of Commands (after initial setup)

```bash
# 0. Ensure you are in project root: C:\Users\FernandoMaytorena\GitHub\vectorfloorseg
# 1. Create and activate virtual environment (if not done)
# python -m venv vectorfloorseg_env
# source vectorfloorseg_env/Scripts/activate

# 2. Upgrade pip
# pip install --upgrade pip

# 3. Install PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install PyTorch Geometric 2.0.4 and related
# pip install torch-geometric==2.0.4
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu118.html

# 5. Install other dependencies (ensure requirements.txt is from VecFloorSeg)
# pip install -r requirements.txt

# 6. Setup custom PyG components (after creating setup_custom_pyg.py)
# python setup_custom_pyg.py

# 7. Verify installation (after creating verify_installation.py)
# python verify_installation.py
```

## Troubleshooting

### Common Issues

1.  **CUDA Version Mismatch**:
    ```bash
    # Check your CUDA version
    nvidia-smi
    # Install matching PyTorch version from pytorch.org
    ```

2.  **PyG Installation Fails**:
    ```bash
    # Try installing without pre-compiled wheels if issues persist
    # pip install torch-geometric==2.0.4 --no-deps
    # pip install torch-scatter torch-sparse --no-index --find-links https://data.pyg.org/whl/torch-1.12.0+cu118.html
    ```

3.  **Custom PyG Setup Fails (`setup_custom_pyg.py`)**:
    *   Ensure you're in the project root directory (`C:\Users\FernandoMaytorena\GitHub\vectorfloorseg`).
    *   Check that the project root contains the `torch_geometric` and `graphgym` folders (these are the custom component sources from the integrated VecFloorSeg codebase).
    *   Verify write permissions to the Python site-packages directory where PyG is installed. This script modifies that installation.
    *   Ensure PyTorch Geometric 2.0.4 was installed correctly before running the script.

4.  **Memory Issues During Installation**:
    ```bash
    # If pip struggles with large packages:
    # pip install --cache-dir /tmp/pip-cache --no-cache-dir <package_name>
    ```

### Verification Checklist

After running `verify_installation.py`, ensure you see:
- ✓ Python version (3.8+)
- ✓ PyTorch with CUDA support (if GPU is intended)
- ✓ PyTorch Geometric 2.0.4 (and its location noted)
- ✓ All required packages (torch_scatter, cv2, numpy, etc.) imported successfully.
- ✓ Confirmation that custom components are assumed integrated if `setup_custom_pyg.py` ran without errors.

## Next Steps

Once Phase 1 is complete and verified:
- Proceed to **Phase 2**: Project Structure Setup and Pretrained Models.
- Commit all new/modified files (`setup_custom_pyg.py`, `verify_installation.py`, updated `phase1_environment_setup.md`, `requirements.txt` if it was created/updated, etc.) to your Git repository.

## Environment Notes

- Always activate the `vectorfloorseg_env` virtual environment before working on the project.
- The custom PyG setup (`setup_custom_pyg.py`) modifies your PyG installation within the virtual environment. Keep the backup it creates if you need to restore the original PyG 2.0.4.
- CUDA version compatibility is critical for GPU performance.
