# -*- coding: utf-8 -*-
"""
Example script for 2D vertical transect plotting
"""

import os
from shyfem.plot.river_plots import RiverTransectPlotter, RiverPlotConfig


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
        time_idx=[1000, 1051],
        y_lims=[-50, 0.5],
        layer='all',
        quiver_n=1,    # Subsample every 3rd node
        quiver_m=1    # Subsample every 10th depth level
    )
    print('stop')
    # # Example 2: Plot with different quiver subsampling
    # plotter.plot_cross_section(
    #     river_branch=base_name,
    #     extracted_folder=extracted_folder,
    #     output_folder=extracted_folder,
    #     nodes_file=nodes_file,
    #     time_idx=0,
    #     y_lims=[-50, 0.5],
    #     layer='all',
    #     quiver_n=5,    # Fewer arrows (every 5th node)
    #     quiver_m=15    # Fewer arrows (every 15th depth)
    # )
    # print('stop')
    # # Example 2: Time range (0 to 9)
    # plotter.plot_cross_section(
    #     river_branch=base_name,
    #     extracted_folder=extracted_folder,
    #     output_folder=extracted_folder,
    #     nodes_file=nodes_file,
    #     time_idx=[0, 9],
    #     y_lims=[-50, 0.5],
    #     layer='all'
    # )
    
    # # Example 3: Surface layer only
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
    """Plot Hovmoller diagram with new features."""
    
    extracted_folder = '/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/sims/NC_out/adrpo_nov_2020'
    base_name = 'line1.dat'
    
    config = RiverPlotConfig(
        plot_ts=True,
        plot_hydro=False,
        time_units='hours since 2020-11-01 00:00:00',
    )
    
    plotter = RiverTransectPlotter(config)
    plotter.load_model_data(extracted_folder, base_name)
    
    # Plot salinity Hovmoller with custom y_lims
    plotter.plot_hovmoller(
        river_branch=base_name,
        output_folder=extracted_folder,
        variable='salinity',
        layer='surface',
        y_lims=[0, 50000],
        time_range=[0, 50]
    )


if __name__ == "__main__":
    print("Generating plots...")
    
    # Run examples
    example_cross_section()
    # example_hovmoller()
    
    print("Done!")  