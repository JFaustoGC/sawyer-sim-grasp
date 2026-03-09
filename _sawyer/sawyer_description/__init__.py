import os

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_urdf_dir() -> str:
    """Returns the absolute path to the urdf directory."""
    return os.path.join(_PACKAGE_ROOT, "urdf")

def get_meshes_dir() -> str:
    """Returns the absolute path to the meshes directory."""
    return os.path.join(_PACKAGE_ROOT, "meshes")

def get_urdf_path(filename: str) -> str:
    """Returns the absolute path to a specific URDF file."""
    return os.path.join(get_urdf_dir(), filename)
