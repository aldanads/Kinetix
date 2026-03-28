"""
Pytest configuration file.
Automatically adds src/ and project root to Python path.
"""
import sys
from pathlib import Path

# Get the project root directory (parent of tests/)
root_path = Path(__file__).parent

# Add src/ to Python path (for imports like: from solvers.base import ...)
src_path = root_path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add project root (for imports like: from utils.mpi_context import ...)
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))