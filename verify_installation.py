# File: verify_installation.py
import sys
import importlib
import os # Added for potentially checking paths if needed

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
