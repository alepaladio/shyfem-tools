## **README.md**

```markdown
# SHYFEM Tools

Post-processing tools for SHYFEM (System of HydrodYnamic Finite Element Modules) model outputs.

## Features

- **Transect Extraction**: Extract nodes along a user-defined line from GRD files
- **NetCDF Processing**: Read, extract, and write SHYFEM NetCDF files with xarray
- **2D Vertical Cross-sections**: Plot salinity, temperature, water level, and velocity along transects
- **Hovmoller Diagrams**: Time vs distance plots for surface/bottom layers
- **2D Horizontal Maps**: Birds-eye view of the domain
- **Video Generation**: Create animations from time series
- **SHY File Support**: Read/write SHYFEM formatted files

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shyfem-tools.git
cd shyfem-tools

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Extract Transect from Shapefile

```python
from shyfem.io import extract_transect_from_files, SHYFEMNodeExtractor

# Extract nodes from GRD using a shapefile line
transect_df = extract_transect_from_files(
    grd_file='path/to/grid.grd',
    shapefile='path/to/transect.shp',
    output_file='transect.dat',
    buffer_distance=7500
)
```

### 2. Extract NetCDF Data Along Transect

```python
# Extract data from SHYFEM NetCDF using the transect file
extractor = SHYFEMNodeExtractor(
    nc_file='path/to/simulation.nc',
    river_file='transect.dat',
    output_dir='./output'
)

output_file = extractor.extract_nodes(
    output_prefix='transect',
    variables=['salinity', 'temperature']
)
```

### 3. Plot 2D Vertical Cross-section

```python
from shyfem.plot.river_plots import RiverTransectPlotter, RiverPlotConfig

config = RiverPlotConfig(
    plot_ts=True,
    plot_hydro=True,
    time_units='hours since 2020-11-01 00:00:00'
)

plotter = RiverTransectPlotter(config)
plotter.load_model_data('./output', 'transect')

plotter.plot_cross_section(
    river_branch='transect',
    extracted_folder='./output',
    nodes_file='transect.dat',
    time_idx=[0, 10],
    y_lims=[-50, 0.5],
    layer='all'
)
```

## Example Workflow

See `shyfem/io/example_usage.py` and `shyfem/plot/example_plotting.py` for complete examples.

## Module Structure

```
shyfem/
├── io/                      # Input/Output operations
│   ├── nc_node_extractor.py # NetCDF extraction
│   └── transect_extractor.py # GRD transect extraction
├── plot/                    # Visualization tools
│   ├── river_plots.py       # 2D vertical cross-sections
│   └── utils.py             # Plotting utilities
└── example_files/           # Example data files
```

## Requirements

- Python >= 3.7
- xarray, netCDF4
- numpy, pandas
- matplotlib, cmocean
- shapely, pyshp

## License

MIT License
```
