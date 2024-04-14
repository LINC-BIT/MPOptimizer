import os
import pathlib


def ensure_dir(file_path: str):
    """Create it if the directory of :attr:`file_path` is not existed.

    Args:
        file_path (str): Target file path.
    """
    
    if isinstance(file_path, pathlib.Path):
        file_path = str(file_path)
    
    if not os.path.isdir(file_path):
        file_path = os.path.dirname(file_path)

    if not os.path.exists(file_path):
        os.makedirs(file_path)
