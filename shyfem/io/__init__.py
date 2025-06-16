"""
Input/Output operations for SHYFEM data formats

Contains:
- NetCDF file readers
- SHY file readers
- Format converters
"""

from .nc_reader import read_nc, get_nc_variables
from .nc_node_extractor import SHYFEMNodeExtractor
from .shy_reader import read_shy_file
from .converters import shy_to_nc, nc_to_shy

__all__ = [
    'read_nc',
    'get_nc_variables',
    'SHYFEMNodeExtractor',
    'read_shy_file',
    'shy_to_nc',
    'nc_to_shy'
]
