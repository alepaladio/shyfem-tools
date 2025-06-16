"""
SHYFEM Tools - A Python library for processing SHYFEM model outputs

Provides tools for:
- Reading SHYFEM NetCDF outputs (3D, 2D-H, 2D-V)
- Converting between SHY and NetCDF formats
- Visualizing hydrodynamic and tracer data
"""

__version__ = "0.1.0"

# Import main components to make available at top level
from .io.nc_node_extractor import SHYFEMNodeExtractor
from .io.converters import shy_to_nc
from .plot.map_plots import plot_2d_horizontal
from .plot.profile_plots import plot_vertical_profile

__all__ = [
    'SHYFEMNodeExtractor',
    'shy_to_nc',
    'plot_2d_horizontal',
    'plot_vertical_profile'
]
