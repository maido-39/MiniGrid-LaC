"""
Multigrid LaC Source Package

This package makes the src/ directory a Python package to easily import lib/ modules.
"""

import sys
from pathlib import Path

# Add src/ directory to Python path
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

