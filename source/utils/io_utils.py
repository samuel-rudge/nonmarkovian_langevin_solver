# source/utils/io_utils.py
"""
Docstring for source.utils.io_utils
"""

from pathlib import Path
import os
import shutil

def clear_directory(dir_path: Path):
    """
    Deletes all files and subdirectories in dir_path, keeping dir_path itself.
    """
    if not dir_path.exists():
        return
    for item in dir_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def ensure_directory(path: Path) -> Path:
    
    path.mkdir(parents=True, exist_ok=True)

    return path


