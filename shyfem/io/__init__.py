"""
Input/Output operations for SHYFEM files
"""

from .nc_node_extractor import SHYFEMNodeExtractor, extract_river_transect
from .transect_extractor import TransectExtractor, extract_transect_from_files
from .shy_reader import SHYReader, read_shy_file
from .shy_writer import SHYWriter, write_shy_file
#from .nc_reader import read_shyfem_nc, standardize_coords
#from .converters import shy_to_netcdf

__all__ = [
    'SHYFEMNodeExtractor',
    'extract_river_transect',
    'TransectExtractor',
    'extract_transect_from_files',
    'SHYReader',
    'read_shy_file',
    'SHYWriter',
    'write_shy_file',
    'read_shyfem_nc',
    'standardize_coords',
    'shy_to_netcdf'
]

#"""
#I/O operations for SHYFEM files
#"""
#
## Import everything from your modules
#from .nc_node_extractor import SHYFEMNodeExtractor, extract_river_transect
#from .shy_reader import SHYReader, read_shy_file
#from .shy_writer import SHYWriter, write_shy_file
#
## List what's available in this submodule
#__all__ = [
#    'SHYFEMNodeExtractor',
#    'extract_river_transect',
#    'SHYReader',
#    'read_shy_file',
#    'SHYWriter',
#    'write_shy_file',
#]