import site
import sys
import os
import shutil
from pathlib import Path

def find_pyg_installation():
    """Find the PyTorch Geometric installation directory in site-packages."""
    # Check standard site-packages locations
    site_packages_dirs = site.getsitepackages()

    # Also check user site-packages, as pip might install there
    user_site_packages = site.getusersitepackages()
    if user_site_packages not in site_packages_dirs:
        site_packages_dirs.append(user_site_packages)
    
    # Also check for virtual environment site-packages
    if hasattr(sys, 'prefix'):
        venv_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
        if venv_site_packages not in site_packages_dirs:
            site_packages_dirs.append(venv_site_packages)

    for sp_dir in site_packages_dirs:
        pyg_path = Path(sp_dir) / "torch_geometric"
        if pyg_path.is_dir():
            # Further check: ensure it's a PyG install, e.g., by checking for a known file/submodule
            if (pyg_path / '__init__.py').exists() and (pyg_path / 'data').is_dir():
                return pyg_path
    return None

def backup_and_replace(src_dir: Path, dest_dir: Path, backup_base_dir: Path):
    """Backs up original components from dest_dir and replaces them with components from src_dir."""
    if not dest_dir.exists() and not dest_dir.is_symlink(): # Allow replacing if dest_dir is a symlink to be overwritten
        # If dest_dir doesn't exist, it might be because we are placing graphgym into a freshly copied custom pyg
        # In this case, we can just copy. But we need to ensure parent exists.
        dest_dir.parent.mkdir(parents=True, exist_ok=True) 
        # print(f"Note: Destination directory {dest_dir} does not exist. Will proceed with copying source.")
    
    # Ensure backup subdirectory for these specific components exists
    # backup_item_specific_dir = backup_base_dir / dest_dir.name # This was causing issues if dest_dir was deep
    # Instead, use a name relative to the component being backed up, e.g. "graphgym_original_within_pyg"
    # The backup_base_dir is already specific enough (e.g. .../torch_geometric_backup_original/graphgym_original_within_pyg)
    # So, backup_base_dir itself can be the parent for items from src_dir.
    
    for item in src_dir.iterdir():
        dest_item_path = dest_dir / item.name
        # backup_item_path = backup_item_specific_dir / item.name # Old logic
        backup_item_path = backup_base_dir / item.name # New logic: backup_base_dir is already specific

        # Ensure backup subdirectory for this item exists
        backup_item_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_item_path.exists() or dest_item_path.is_symlink():
            # Backup existing item (file or directory)
            if dest_item_path.is_dir() and not dest_item_path.is_symlink():
                if not backup_item_path.exists(): # Avoid re-backing up if script is run multiple times
                    shutil.copytree(dest_item_path, backup_item_path, dirs_exist_ok=True, symlinks=True)
                shutil.rmtree(dest_item_path) # Remove original directory
            elif dest_item_path.is_file() or dest_item_path.is_symlink(): # It's a file or symlink
                if not backup_item_path.exists():
                    # For symlinks, copy2 would copy the target. We want to preserve the link if it's a link.
                    # However, the goal here is to backup what *was* there. If it was a link, backup the link.
                    # If it was a file, backup the file.
                    # Since copy2 on a link copies the target file, we need to handle links differently for backup.
                    # But for simplicity and given the context (replacing with project files),
                    # backing up the content (if link) or file is acceptable.
                    # If dest_item_path is a symlink, os.remove works.
                    if dest_item_path.is_symlink():
                        # To "backup" a symlink, we'd record its target. Here, we just remove it.
                        # The new item will replace it. If backup_item_path is to store the link itself,
                        # that's more complex. For now, we assume we backup the *content* if it's a link to a file/dir.
                        # Let's assume for now that if it's a symlink, we are interested in what it points to for backup,
                        # or simply removing it to replace.
                        # The original shutil.copy2(dest_item_path, backup_item_path) would copy the *target* of the symlink.
                        # This might be okay.
                        pass # Symlink will be removed next.
                    
                    shutil.copy2(dest_item_path, backup_item_path, follow_symlinks=False) # copy2 preserves metadata
                os.remove(dest_item_path) # Remove original file or symlink
            print(f"  Backed up '{dest_item_path.name}' to '{backup_item_path}'")

        # Copy new item
        if item.is_dir():
            shutil.copytree(item, dest_item_path, dirs_exist_ok=True, symlinks=True)
        else: # File or symlink from source
            if item.is_symlink():
                # If source is a symlink, copy it as a symlink
                link_to = os.readlink(item)
                os.symlink(link_to, dest_item_path)
            else:
                shutil.copy2(item, dest_item_path) # copy2 preserves metadata
        print(f"  Replaced '{dest_item_path.name}' with custom version from '{item}'")


def main():
    pyg_install_path = find_pyg_installation()
    if not pyg_install_path:
        print("Error: PyTorch Geometric installation not found. Please ensure PyG 2.0.4 is installed.")
        sys.exit(1)
    
    print(f"Found PyTorch Geometric installation at: {pyg_install_path}")

    project_root = Path(__file__).parent.resolve()
    CUSTOM_PYG_SOURCE_DIR = project_root / "torch_geometric" 
    CUSTOM_GRAPHGYM_SOURCE_DIR = project_root / "graphgym"   

    pyg_parent_dir = pyg_install_path.parent
    backup_dir_base = pyg_parent_dir / "vectorfloorseg_backup_originals" # Renamed for clarity
    
    print(f"Original components will be backed up under: {backup_dir_base}")

    # 1. Handle PyTorch Geometric custom components
    if CUSTOM_PYG_SOURCE_DIR.exists() and CUSTOM_PYG_SOURCE_DIR.is_dir():
        print(f"\nProcessing custom PyTorch Geometric components from: {CUSTOM_PYG_SOURCE_DIR}")
        
        original_pyg_full_backup_path = backup_dir_base / "torch_geometric_full_original"
        if pyg_install_path.exists(): # Check if the original PyG path still exists
            if not original_pyg_full_backup_path.exists():
                original_pyg_full_backup_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Backing up original 'torch_geometric' directory from '{pyg_install_path}' to '{original_pyg_full_backup_path}'...")
                shutil.copytree(pyg_install_path, original_pyg_full_backup_path, dirs_exist_ok=True, symlinks=True)
                print(f"  Successfully backed up original 'torch_geometric' directory.")
            else:
                print(f"  Original 'torch_geometric' directory already backed up at '{original_pyg_full_backup_path}'. Skipping backup.")
            
            print(f"  Removing original 'torch_geometric' directory from '{pyg_install_path}'...")
            shutil.rmtree(pyg_install_path)
            print(f"  Successfully removed original 'torch_geometric' directory.")
        else:
            print(f"  Original 'torch_geometric' directory at '{pyg_install_path}' not found. Assuming it was already removed or replaced.")

        print(f"  Copying custom 'torch_geometric' directory from '{CUSTOM_PYG_SOURCE_DIR}' to '{pyg_install_path}'...")
        shutil.copytree(CUSTOM_PYG_SOURCE_DIR, pyg_install_path, dirs_exist_ok=True, symlinks=True)
        print(f"  Successfully copied custom 'torch_geometric' directory.")
        
    else:
        print(f"Error: Custom PyG source directory not found: {CUSTOM_PYG_SOURCE_DIR}")
        print("Ensure the 'torch_geometric' directory (from VecFloorSeg) is at the project root.")
        sys.exit(1)

    # 2. Handle GraphGym custom components
    pyg_graphgym_dest_path = pyg_install_path / "graphgym" 

    if CUSTOM_GRAPHGYM_SOURCE_DIR.exists() and CUSTOM_GRAPHGYM_SOURCE_DIR.is_dir():
        print(f"\nProcessing custom GraphGym components from: {CUSTOM_GRAPHGYM_SOURCE_DIR}")
        # Specific backup location for graphgym components that were inside the original PyG
        backup_graphgym_specific_dir = backup_dir_base / "graphgym_original_within_pyg"
        
        # The backup_and_replace function handles sub-components.
        # It will back up items from pyg_graphgym_dest_path (if they exist) into backup_graphgym_specific_dir,
        # then copy items from CUSTOM_GRAPHGYM_SOURCE_DIR to pyg_graphgym_dest_path.
        print(f"  Custom GraphGym components will be placed into: {pyg_graphgym_dest_path}")
        print(f"  Original GraphGym components (if any within PyG) will be backed up under: {backup_graphgym_specific_dir}")
        backup_and_replace(CUSTOM_GRAPHGYM_SOURCE_DIR, pyg_graphgym_dest_path, backup_graphgym_specific_dir)
    else:
        print(f"\nWarning: Custom GraphGym source directory '{CUSTOM_GRAPHGYM_SOURCE_DIR}' not found.")
        print("Ensure the 'graphgym' directory (from VecFloorSeg) is at the project root if custom GraphGym components are needed.")
        print("Proceeding without custom GraphGym components if not found or not applicable.")

    print("\nCustom PyG and GraphGym setup complete.")
    print(f"Original files are backed up in subdirectories under: {backup_dir_base}")
    print("To restore, copy backed-up 'torch_geometric_full_original' back to site-packages, replacing the custom one.")
    print("And similarly for GraphGym if it was customized (from 'graphgym_original_within_pyg').")

if __name__ == "__main__":
    main()
