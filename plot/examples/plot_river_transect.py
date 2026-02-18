"""
Example script for plotting river transects
"""

from shyfem.plot.river_plots import RiverTransectPlotter, RiverPlotConfig

# Configuration
config = RiverPlotConfig(
    plot_hydro=False,
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

# Create plotter
plotter = RiverTransectPlotter(config)

# Load data
models_folder = '/path/to/your/nc/files'
river_branch = 'PoDiVenezia'
sims_name = 'deltapo_ER_202207_ogridNoDRCVmin'

plotter.load_discharge_data('/path/to/discharge_po_pontelagoscuro.dat')
plotter.load_model_data(models_folder, river_branch)

# Generate plots
output_folder = models_folder
plotter.plot(river_branch, sims_name, output_folder)