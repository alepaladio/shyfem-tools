"""
SHYFEM Tools - Post-processing tools for SHYFEM model outputs
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import key classes/functions for top-level access
# ONLY import things that actually exist!

# Import from io module
from .io.nc_node_extractor import SHYFEMNodeExtractor, extract_river_transect

# List what will be available when someone does: from shyfem import *
__all__ = [
    'SHYFEMNodeExtractor',
    'extract_river_transect',
]