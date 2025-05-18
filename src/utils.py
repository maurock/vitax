"""Utils for the project."""
import os


def make_dir(path: str) -> None:
    """Make a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")