# File: download_backbone.py
import urllib.request
import os
from pathlib import Path
import torch
import hashlib

def verify_file_hash(filepath: str, expected_hash: str) -> bool:
    """Verify downloaded file integrity using SHA256 hash.

    Args:
        filepath (str): Path to the file.
        expected_hash (str): Expected SHA256 hash.

    Returns:
        bool: True if hash matches, False otherwise.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash

def download_resnet_backbone() -> bool:
    """Download the required ResNet-101 pretrained weights.

    Returns:
        bool: True if download and verification are successful, False otherwise.
    """
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ResNet-101 pretrained on ImageNet
    resnet_url = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
    resnet_path = models_dir / "resnet101-torch.pth"
    # Note: verify_file_hash is defined but not used for resnet101 in the original guide's snippet.
    # If a full SHA256 hash is available, verify_file_hash could be integrated.
    
    if not resnet_path.exists():
        print("Downloading ResNet-101 pretrained weights...")
        try:
            urllib.request.urlretrieve(resnet_url, resnet_path)
            print(f"✓ Downloaded ResNet-101 to: {resnet_path}")
        except Exception as e:
            print(f"✗ Failed to download ResNet-101: {e}")
            return False
    else:
        print(f"✓ ResNet-101 weights already exist at: {resnet_path}")
    
    # Verify the model can be loaded
    try:
        checkpoint = torch.load(resnet_path, map_location='cpu')
        num_items = len(checkpoint.keys()) if isinstance(checkpoint, dict) else len(checkpoint)
        print(f"✓ ResNet-101 model verified - {num_items} items in state_dict")
        return True
    except Exception as e:
        print(f"✗ Failed to load ResNet-101 model: {e}")
        return False

def download_additional_backbones() -> None:
    """Download additional backbone options (ResNet-50, VGG-16)."""
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True) # Ensure models_dir exists
    
    # ResNet-50 (lighter alternative)
    resnet50_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    resnet50_path = models_dir / "resnet50-torch.pth"
    
    if not resnet50_path.exists():
        print("Downloading ResNet-50 pretrained weights...")
        try:
            urllib.request.urlretrieve(resnet50_url, resnet50_path)
            print(f"✓ Downloaded ResNet-50 to: {resnet50_path}")
        except Exception as e:
            print(f"✗ Failed to download ResNet-50: {e}")
    else:
        print(f"✓ ResNet-50 weights already exist at: {resnet50_path}")

    # VGG-16 (another option)
    vgg16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
    vgg16_path = models_dir / "vgg16-torch.pth"

    if not vgg16_path.exists():
        print("Downloading VGG-16 pretrained weights...")
        try:
            urllib.request.urlretrieve(vgg16_url, vgg16_path)
            print(f"✓ Downloaded VGG-16 to: {vgg16_path}")
        except Exception as e:
            print(f"✗ Failed to download VGG-16: {e}")
    else:
        print(f"✓ VGG-16 weights already exist at: {vgg16_path}")

if __name__ == "__main__":
    print("=== Downloading Pretrained Backbone Models ===")
    
    resnet_success = download_resnet_backbone()
    
    if resnet_success:
        print("\n=== Downloading Additional Backbones ===")
        download_additional_backbones()
        print("\n✓ All backbone downloads attempted.")
    else:
        print("\n✗ ResNet-101 download/verification failed. Additional backbones were not downloaded.")
