# -*- coding: utf-8 -*-
"""
Example script for 2D vertical transect plotting
"""

import os
from shyfem.plot.river_plots import RiverTransectPlotter, RiverPlotConfig
import matplotlib.pyplot as plt

def example_cross_section():
    """Plot 2D vertical cross-section with all new features."""
    
    extracted_folder = '/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/sims/NC_out/adrpo_nov_2020'
    base_name = 'line1.dat'
    
    # Path to nodes file (optional)
    nodes_file = '/home/utente/Documenti/aph_shyfem_tools/shyfem-tools/shyfem/plot/line1.dat'
    
    # Set overall configuration
    config = RiverPlotConfig(
        plot_hydro=True,
        plot_ts=True,
        time_units='hours since 2020-11-01 00:00:00'
        # x_lim=(0, 50000),
        # y_lim=(-50, 1)
    )
    plotter = RiverTransectPlotter(config)
    plotter.load_model_data(extracted_folder, base_name)
    
    # Example 1: Single time step with custom y_lims and quivers
    plotter.plot_cross_section(
        river_branch=base_name,
        extracted_folder=extracted_folder,
        output_folder=extracted_folder,
        nodes_file=nodes_file,
        time_idx=[1000, 1011],
        y_lims=[-50, 0.5],
        layer='all',
        quiver_n=1,    # Subsample every 3rd node
        quiver_m=1,    # Subsample every 10th depth level
        make_video=False,
        fps=4

    )
    print('Finished example 1.')
    # Example 3: Surface layer only
    # plotter.plot_cross_section(
    #     river_branch=base_name,
    #     extracted_folder=extracted_folder,
    #     output_folder=extracted_folder,
    #     nodes_file=nodes_file,
    #     time_idx=0,
    #     y_lims=[-50, 0.5],
    #     layer='surface'
    # )
    
    # # Example 4: All time steps
    # plotter.plot_cross_section(
    #     river_branch=base_name,
    #     extracted_folder=extracted_folder,
    #     output_folder=extracted_folder,
    #     nodes_file=nodes_file,
    #     time_idx='all',
    #     y_lims=[-50, 0.5],
    #     layer='all'
    # )


def example_hovmoller():
    """Plot Hovmoller diagram."""
    
    extracted_folder = '/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/sims/NC_out/adrpo_nov_2020'
    base_name = 'line1.dat'
    
    config = RiverPlotConfig(
        plot_ts=True,
        plot_hydro=False,
        time_units='hours since 2020-11-01 00:00:00',
    )
    
    plotter = RiverTransectPlotter(config)
    plotter.load_model_data(extracted_folder, base_name)
    
    # Hovmoller with threshold for open sea (35 g/kg)
    plotter.plot_hovmoller(
        river_branch=base_name,
        output_folder=extracted_folder,
        variable='salinity',
        layer='bottom',
        threshold=29,
        x_tick_space=30,
        time_range=[0, 1000]
    )

    # Hovmoller for river branch (2 g/kg threshold)
    plotter.plot_hovmoller(
        river_branch=base_name,
        output_folder=extracted_folder,
        variable='salinity',
        layer='bottom',
        x_tick_space=30,
        threshold=28,
    )

    # Surface layer without threshold
    plotter.plot_hovmoller(
        river_branch=base_name,
        output_folder=extracted_folder,
        variable='salinity',
        layer='surface',
        x_tick_space=40
    )
    print('done ex. 3')
    
def example_horizontal_maps():
    """
    Example of 2D horizontal maps with explicit file paths.
    """
    
    # ============================================================
    # PATHS - Explicitly point to your files
    # ============================================================
    
    # Full paths to your NetCDF files
    ts_file = '/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/sims/adrpo_nov_2020_ts.nc'
    hydro_file = '/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/sims/adrpo_nov_2020_hydro.nc'
    
    # GRD file for triangulation
    grd_file = '/home/utente/Documenti/aph_shyfem_tools/shyfem-tools/shyfem/example_files/def7_geo.grd'
    
    # Shapefile for overlay (optional)
    shapefile = '/home/utente/Documenti/OMBRES/QGIS/line_along_adriatic.shp'
    
    # DAT files to overlay
    dat_files = ['/home/utente/Documenti/aph_shyfem_tools/shyfem-tools/shyfem/plot/line1.dat']
    
    # Output folder for plots
    output_folder = '/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/sims/NC_out/adrpo_nov_2020'
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    config = RiverPlotConfig(
        plot_ts=True,
        plot_hydro=True,
        time_units='hours since 2020-11-01 00:00:00'
    )
    
    plotter = RiverTransectPlotter(config)
    
    # Shapefiles to overlay
    shapefiles = [
        {'path': shapefile, 'type': 'line'}
    ]
    
    # ============================================================
    # EXAMPLE 1: TS only (salinity + temperature) - Surface layer
    # ============================================================
    print("\nExample 1: TS only - Surface")
    plotter.plot_map(
        river_branch='adrpo_nov_2020',
        grd_file=grd_file,
        ts_file=ts_file,
        hydro_file=hydro_file,
        output_folder=output_folder,
        time_idx=[1000, 1010],
        x_lims=[11.786, 16.546],
        y_lims=[43.113, 45.867],
        layer='bottom',  # or layer=0
        plot_ts=True,
        plot_hydro=True,
        plot_quivers=True,
        quiver_grid_resolution=50,
        shapefiles=shapefiles,
        dat_files=dat_files,
        save_plot=True,
        make_video=False
    )
    
    # ============================================================
    # EXAMPLE 2: TS only - Bottom layer with custom limits
    # ============================================================
    print("\nExample 2: TS only - Bottom layer")
    plotter.plot_map(
        river_branch='adrpo_nov_2020',
        grd_file=grd_file,
        ts_file=ts_file,
        hydro_file=None,
        output_folder=output_folder,
        time_idx=1000,
        x_lims=[11.786, 16.546],
        y_lims=[43.113, 45.867],
        layer='bottom',
        plot_ts=True,
        plot_hydro=False,
        plot_quivers=False,
        shapefiles=shapefiles,
        save_plot=True,
        make_video=False,
        sal_limits=[30, 38],  # Custom salinity limits
        temp_limits=[12, 20]  # Custom temperature limits
    )
    
    # ============================================================
    # EXAMPLE 3: Specific layer index (layer 5)
    # ============================================================
    print("\nExample 3: Layer 5")
    plotter.plot_map(
        river_branch='adrpo_nov_2020',
        grd_file=grd_file,
        ts_file=ts_file,
        hydro_file=hydro_file,
        output_folder=output_folder,
        time_idx=1000,
        x_lims=[11.786, 16.546],
        y_lims=[43.113, 45.867],
        layer=5,  # Specific layer index
        plot_ts=True,
        plot_hydro=True,
        plot_quivers=True,
        shapefiles=shapefiles,
        save_plot=True,
        make_video=False,
        sal_limits=[35, 38],
        wl_limits=[-0.5, 0.5],
        vel_limits=[0, 0.5]
    )

    
if __name__ == "__main__":
    print("Generating plots...")
    
    # Run examples
    example_cross_section()
    # example_hovmoller()
    # example_horizontal_maps()
    # example_map_with_basemap()
    plt.close()
    print("Done!")  