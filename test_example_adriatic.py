
## **Combined Example: `shyfem/plot/example_workflow.py`**

```python
"""
Complete workflow: Extract transect from shapefile -> NetCDF -> Plot
"""

import os
import xarray as xr
from shyfem.io import extract_transect_from_files, SHYFEMNodeExtractor
from shyfem.plot.river_plots import RiverTransectPlotter, RiverPlotConfig


def example_complete_workflow():
    """
    Complete workflow: 
    1. Extract transect nodes from GRD using shapefile
    2. Extract NetCDF data along the transect
    3. Plot 2D vertical cross-sections
    """
    
    # ============================================================
    # STEP 1: Extract transect nodes from GRD
    # ============================================================
    
    GRD_FILE = "./example_files/def7_geo.grd"
    SHAPEFILE = "./io/example_files/line_along_adriatic.shp"
    DAT_FILE = "./plot/line1.dat"
    
    print("Step 1: Extracting transect from GRD...")
    transect_df = extract_transect_from_files(
        grd_file=GRD_FILE,
        shapefile=SHAPEFILE,
        output_file=DAT_FILE,
        buffer_distance=7500,
        start_point_id=0
    )
    
    if transect_df is None:
        print("Failed to extract transect")
        return
    
    print(f"Transect extracted: {len(transect_df)} nodes")
    
    # ============================================================
    # STEP 2: Extract NetCDF data along transect
    # ============================================================
    
    main_folder = '/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/sims'
    sim_name = 'adrpo_nov_2020'
    output_folder = f'{main_folder}/NC_out/{sim_name}'
    os.makedirs(output_folder, exist_ok=True)
    
    print("\nStep 2: Extracting NetCDF data...")
    
    for varid in ["hydro", "ts"]:
        filename = f'{sim_name}_{varid}.nc'
        
        extractor = SHYFEMNodeExtractor(
            nc_file=f"{main_folder}/{filename}",
            river_file=DAT_FILE,
            output_dir=output_folder
        )
        
        if varid == "ts":
            variables = ["salinity", "temperature"]
        else:
            variables = ["water_level", "u_velocity", "v_velocity"]
        
        output_file = extractor.extract_nodes(
            output_prefix="line1",
            sort_direction="longitude",
            save_frequency=1,
            variables=variables,
            nc_file_varid=varid
        )
        
        ds = xr.open_dataset(output_file)
        print(f"  {varid}: {list(ds.data_vars)}")
        extractor.close()
    
    # ============================================================
    # STEP 3: Plot cross-sections
    # ============================================================
    
    print("\nStep 3: Generating plots...")
    
    config = RiverPlotConfig(
        plot_hydro=True,
        plot_ts=True,
        time_units='hours since 2020-11-01 00:00:00'
    )
    
    plotter = RiverTransectPlotter(config)
    plotter.load_model_data(output_folder, "line1")
    
    # Plot multiple time steps
    plotter.plot_cross_section(
        river_branch="line1",
        extracted_folder=output_folder,
        output_folder=output_folder,
        nodes_file=DAT_FILE,
        time_idx=[1000, 1051],
        y_lims=[-50, 0.5],
        layer='all'
    )
    
    print("\nWorkflow complete!")
    print(f"Outputs saved to: {output_folder}")
    print(f"Plots saved to: {output_folder}/line1_plots/cross_section/")


if __name__ == "__main__":
    example_complete_workflow()
