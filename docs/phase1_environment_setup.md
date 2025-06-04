# VectorFloorSeg Implementation - Phase 1: Environment Setup and Dependencies

## Overview

This document covers the complete environment setup for VectorFloorSeg using pip and virtual environments instead of conda. The main challenge is handling the custom PyTorch Geometric modifications that the project requires.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git for cloning repositories

## 1.1 Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv vectorfloorseg_env

# Activate the environment
# On Linux/Mac:
source vectorfloorseg_env/bin/activate
# On Windows:
vectorfloorseg_env\Scripts\activate
```

## 1.2 Clone Repository and Install Base Dependencies

```bash
# Clone the VectorFloorSeg repository
git clone https://github.com/DrZiji/VecFloorSeg.git
cd VecFloorSeg

# Upgrade pip to latest version
pip install --upgrade pip

# Install PyTorch first (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 1.3 Install PyTorch Geometric 2.0.4

The project specifically requires PyG 2.0.4 with custom modifications:

```bash
# Install PyTorch Geometric 2.0.4 and related packages
pip install torch-geometric==2.0.4
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu118.html
```

## 1.4 Handle Custom PyG Modifications

The critical step - replacing standard PyG components with custom ones:

```python
# File: setup_custom_pyg.py
import os
import shutil
import site
import sys
from pathlib import Path

def find_pyg_installation():
    """Find the PyTorch Geometric installation directory."""
    site_packages = site.getsitepackages()
    
    for sp in site_packages:
        pyg_path = Path(sp) / "torch_geometric"
        if pyg_path.exists():
            return pyg_path
    
    # Try user site-packages as fallback
    user_site = site.getusersitepackages()
    pyg_path = Path(user_site) / "torch_geometric"
    if pyg_path.exists():
        return pyg_path
    
    raise FileNotFoundError("PyTorch Geometric installation not found")

def setup_custom_pyg():
    """Replace standard PyG components with VectorFloorSeg custom versions."""
    
    # Find PyG installation
    pyg_install_path = find_pyg_installation()
    print(f"Found PyG installation at: {pyg_install_path}")
    
    # Paths to custom components
    repo_root = Path(__file__).parent
    custom_torch_geometric = repo_root / "torch_geometric"
    custom_graphgym = repo_root / "graphgym"
    
    # Backup original files
    backup_dir = repo_root / "pyg_backup"
    backup_dir.mkdir(exist_ok=True)
    
    if not (backup_dir / "torch_geometric").exists():
        print("Creating backup of original PyG installation...")
        shutil.copytree(pyg_install_path, backup_dir / "torch_geometric")
    
    # Replace torch_geometric components
    if custom_torch_geometric.exists():
        print("Replacing torch_geometric components...")
        for item in custom_torch_geometric.iterdir():
            dest = pyg_install_path / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    
    # Handle graphgym - this might need to be added to PyG
    graphgym_dest = pyg_install_path / "graphgym"
    if custom_graphgym.exists():
        print("Adding custom graphgym to PyG...")
        if graphgym_dest.exists():
            shutil.rmtree(graphgym_dest)
        shutil.copytree(custom_graphgym, graphgym_dest)
    
    print("Custom PyG setup complete!")

if __name__ == "__main__":
    setup_custom_pyg()
```

## 1.5 Install Additional Dependencies

```bash
# Install other required packages
pip install opencv-python
pip install matplotlib
pip install numpy
pip install scipy
pip install Pillow
pip install tqdm
pip install tensorboard
pip install wandb  # for experiment tracking
pip install albumentations  # for data augmentation

# Install packages for SVG processing
pip install svglib
pip install cairosvg
pip install xml.etree.ElementTree  # usually built-in

# Install mmsegmentation components (if needed)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch1.12.0/index.html
pip install mmsegmentation
```

## 1.6 Verify Installation

```python
# File: verify_installation.py
import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np

def verify_pytorch_installation():
    """Verify PyTorch and CUDA setup."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    return torch.cuda.is_available()

def verify_pyg_installation():
    """Verify PyTorch Geometric installation and custom components."""
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    
    # Test basic PyG functionality
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    print(f"Created test graph with {data.num_nodes} nodes and {data.num_edges} edges")
    
    # Check for custom components
    try:
        from torch_geometric.nn import GATConv
        print("✓ Standard GAT layer available")
    except ImportError as e:
        print(f"✗ GAT import error: {e}")
    
    # Try to import custom modulated GAT (if available)
    try:
        # This will depend on the custom implementation
        print("✓ Custom components appear to be installed")
    except ImportError:
        print("⚠ Custom components may not be fully integrated")

def verify_other_dependencies():
    """Verify other required packages."""
    packages = [
        'cv2', 'matplotlib', 'numpy', 'scipy', 'PIL',
        'tqdm', 'tensorboard'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError:
            print(f"✗ {package} import failed")

if __name__ == "__main__":
    print("=== Verifying VectorFloorSeg Installation ===")
    
    cuda_available = verify_pytorch_installation()
    print("\n" + "="*50)
    
    verify_pyg_installation()
    print("\n" + "="*50)
    
    verify_other_dependencies()
    print("\n" + "="*50)
    
    if cuda_available:
        print("✓ Installation verification complete - GPU acceleration available")
    else:
        print("⚠ Installation complete but no GPU acceleration (CPU only)")
```

## Installation Sequence

Run these commands in order:

```bash
# 1. Create and activate environment
python -m venv vectorfloorseg_env
source vectorfloorseg_env/bin/activate  # Linux/Mac
# vectorfloorseg_env\Scripts\activate  # Windows

# 2. Clone repository and install PyTorch
git clone https://github.com/DrZiji/VecFloorSeg.git
cd VecFloorSeg
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric
pip install torch-geometric==2.0.4
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu118.html

# 4. Install additional dependencies
pip install opencv-python matplotlib numpy scipy Pillow tqdm tensorboard wandb albumentations svglib cairosvg

# 5. Setup custom PyG components
python setup_custom_pyg.py

# 6. Verify installation
python verify_installation.py
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**:
   ```bash
   # Check your CUDA version
   nvidia-smi
   # Install matching PyTorch version
   ```

2. **PyG Installation Fails**:
   ```bash
   # Try installing without pre-compiled wheels
   pip install torch-geometric==2.0.4 --no-deps
   pip install torch-scatter torch-sparse --no-index --find-links https://data.pyg.org/whl/torch-1.12.0+cu118.html
   ```

3. **Custom PyG Setup Fails**:
   - Ensure you're in the VecFloorSeg directory
   - Check that the repository contains `torch_geometric` and `graphgym` folders
   - Verify write permissions to the site-packages directory

4. **Memory Issues During Installation**:
   ```bash
   # Increase pip cache and use fewer parallel jobs
   pip install --cache-dir /tmp/pip-cache --no-cache-dir
   ```

### Verification Checklist

After running `verify_installation.py`, ensure you see:
- ✓ PyTorch with CUDA support
- ✓ PyTorch Geometric 2.0.4
- ✓ All required packages imported successfully
- ✓ Custom components integrated (if repository contains them)

## Next Steps

Once Phase 1 is complete, proceed to:
- **Phase 2**: Project Structure Setup and Pretrained Models
- **Phase 3**: Data Preparation Pipeline
- **Phase 4**: Model Implementation
- **Phase 5**: Training Script

## Environment Notes

- Always activate the virtual environment before working on the project
- The custom PyG setup modifies your PyG installation - keep the backup in case you need to restore
- CUDA version compatibility is critical for performance
- Consider using `conda` if you encounter persistent dependency conflicts, though this guide should handle most cases
