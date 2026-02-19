# -*- coding: utf-8 -*-
"""
Example script for plotting river transects
"""

import os
# from shyfem.io.nc_node_extractor import SHYFEMNodeExtractor, extract_river_transect
# import xarray as xr
# import matplotlib.pyplot as plt
from shyfem.plot.river_plots import RiverTransectPlotter, RiverPlotConfig

# Create config
config = RiverPlotConfig(
    plot_hydro=True,
    plot_ts=True,
    plot_measurements=False,
    make_video=True,
    x_lim=(0, 50000),
    y_lim=(-9.5, 1),
    max_discharge_lim=750,
    time_units='hours since 2022-07-15 00:00:00',
    time_calendar='standard',
    date_range=("2022-07-01", "2022-09-01")
)

# Main folder:
models = '/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/NC_out/deltapo_ER_202207_ogridNoDRCVmin_WATEXT_f'
river_branch = 'PoDiVenezia'
filename_discharge = '/home/utente/Documenti/climaxpo/discharge_po_pontelagoscuro.dat'

# Create plotter instance
plotter = RiverTransectPlotter(config)
# Configuration


# Create plotter
plotter = RiverTransectPlotter(config)

# Load data
models_folder = models
river_branch = 'PoDiVenezia'
sims_name = 'deltapo_ER_202207_ogridNoDRCVmin'

plotter.load_discharge_data(filename_discharge)
plotter.load_model_data(models_folder, river_branch)

# Generate plots
output_folder = models_folder
plotter.plot(river_branch, sims_name, output_folder)
