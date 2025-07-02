import os
import shutil

def delete_pycache_folders(root_folder):
    deleted = 0
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Deleted: {pycache_path}")
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {pycache_path}: {e}")
    print(f"\nTotal __pycache__ folders deleted: {deleted}")

# Replace 'genesis' with the full or relative path to your folder
delete_pycache_folders(r"C:\Users\Zaur\OneDrive\Development\Genesis\genesis")