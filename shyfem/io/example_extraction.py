"""
Example usage of SHYFEMNodeExtractor with xarray
"""

import os
from shyfem.io.nc_node_extractor import SHYFEMNodeExtractor, extract_river_transect
from shyfem.io import extract_transect_from_files, SHYFEMNodeExtractor
import xarray as xr
import matplotlib.pyplot as plt

# Example 0: Transect extraction from a shapefile to a netCDF file
def example_transect_from_shapefile():
        """Complete workflow: extract transect from GRD, then extract NC data"""
        # ========== STEP 1: Extract transect nodes from GRD ==========
        # Input files
        # GRD_FILE = "/home/utente/Documenti/OMBRES/grid_ff/adri_lags_15mPiles_276714_excluded.grd"
        GRD_FILE = "/home/utente/Documenti/shyfem_wiki/adriatic_po_2020/grid/def7_geo.grd"
        SHAPEFILE = "/home/utente/Documenti/OMBRES/QGIS/line_along_adriatic.shp"
        
        # Output file (same name as shapefile but .dat)
        # output_name = os.path.splitext(os.path.basename(SHAPEFILE))[0] + ".dat"
        output_name = 'line1.dat'
        DAT_FILE = os.path.join(os.path.dirname(SHAPEFILE), output_name)
        
        # Extract transect from GRD
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
        print(transect_df.head())
        
        # ========== STEP 2: Use the transect file with SHYFEMNodeExtractor ==========
        
        # Setup for NC extraction
        main_folder = ''
        varid = 'hydro'
        sim_name = ''
        filename = f'{sim_name}_{varid}.nc'
        output_folder = f'{main_folder}/NC_out/{sim_name}'
        os.makedirs(output_folder, exist_ok=True)
        
        # Use the DAT file as river_file input
        riverid = os.path.splitext(os.path.basename(SHAPEFILE))[0]  # Use shapefile name
        
        extractor = SHYFEMNodeExtractor(
            nc_file=f"{main_folder}/{filename}",
            river_file=DAT_FILE,  # The DAT file we just created
            output_dir=f"{output_folder}"
        )
        
        # Extract nodes along transect
        if varid == "ts":
            variables = ["salinity", "temperature"]
        else:
            variables = ["water_level", "u_velocity", "v_velocity"]
        
        output_file = extractor.extract_nodes(
            output_prefix=f"{riverid}",
            sort_direction="longitude",  # Order based on lon/lat from DAT file
            save_frequency=1,
            variables=variables,
            nc_file_varid=varid
        )
        
        # Load and inspect the extracted data
        ds_extracted = xr.open_dataset(output_file)
        print(f"Extracted dataset dimensions: {ds_extracted.dims}")
        print(f"Variables: {list(ds_extracted.data_vars)}")
        
        extractor.close()
        return ds_extracted

# Example 1: Basic extraction
def example_basic():
    """Basic example of extracting river nodes."""
    
    # Initialize extractor
    # extractor = SHYFEMNodeExtractor(
    #     nc_file="/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/deltapo_ER_202207_ogridNoDRCVmin_WATEXT_ts_uns.nc",
    #     river_file="/home/utente/Documenti/climaxpo/qgis/river_branches/nodes_PoDiVeneziaF.csv",
    #     output_dir="/home/utente/Documenti/climaxpo/climaxpo_2022-2023/test_results"
    # )
    # Select NC file to extract data from 
    main_folder = '/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207'
    # varid = 'ts' # ts or hydro
    varid = 'hydro' # ts or hydro
    # sim_name = 'deltapo_ER_202207_ogridNoDRCVmin'
    sim_name = 'deltapo_ER_202207_ogridNoDRCVmin_WATEXT_f'
    # filename = f'{sim_name}_{varid}_20220715-20220725.nc'
    filename = f'{sim_name}_{varid}_20220701-20220801.nc'
    output_folder = f'{main_folder}/NC_out/{sim_name}'
    os.makedirs(output_folder, exist_ok=True)
    # deltapo_ER_202207_ogridNoDRCVmin_ts_20220715-20220725.nc
    # deltapo_ER_202207_ogridNoDRCVmin_hydro_20220715-20220725.nc
    
    # River to extract
    riverid = 'PoDiVenezia'
    # riverid = 'PoDiGoro'
    extractor = SHYFEMNodeExtractor(
        nc_file=f"{main_folder}/{filename}",
        river_file=f"/home/utente/Documenti/climaxpo/qgis/river_branches/nodes_{riverid}F.csv",
        output_dir=f"{output_folder}"
    )
    
    # Extract nodes along river (west-east sorting)
    if varid == "ts":
        variables = ["salinity", "temperature"]
    else:
        variables = ["water_level", "u_velocity", "v_velocity"]
    
    output_file = extractor.extract_nodes(
        output_prefix=f"{riverid}",
        sort_direction="longitude",
        save_frequency=1,
        variables=variables,
        nc_file_varid=varid
    )
    # variables=['salinity', 'temperature']  # Optional: select specific variables
    # variables=['water_level', 'zeta', 'u_velocity', 'v_velocity']
    
    # Load and inspect the extracted data
    ds_extracted = xr.open_dataset(output_file)
    print(f"Extracted dataset dimensions: {ds_extracted.dims}")
    print(f"Variables: {list(ds_extracted.data_vars)}")
    
    # Plot a quick visualization
    if 'salinity' in ds_extracted:
        ds_extracted.salinity.isel(time=0).plot()
        plt.title("Salinity along transect (first time step)")
        plt.savefig("salinity_transect.png", dpi=150, bbox_inches='tight')
    
    extractor.close()
    return ds_extracted


# Example 2: Using convenience function
def example_convenience():
    """Using the convenience function for quick extraction."""
    
    output_file = extract_river_transect(
        nc_file="/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/deltapo_ER_202207_ogridNoDRCVmin_WATEXT_ts_uns.nc",
        river_file="/home/utente/Documenti/climaxpo/qgis/river_branches/nodes_PoDiVeneziaF.csv",
        sort_direction="longitude",
        save_frequency=1
    )
    
    print(f"Created: {output_file}")
    return output_file


# Example 3: Extract arbitrary transect (not just river)
def example_arbitrary_transect():
    """Extract data along an arbitrary line between two points."""
    
    extractor = SHYFEMNodeExtractor(
        nc_file="/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/deltapo_ER_202207_ogridNoDRCVmin_WATEXT_ts_uns.nc",
    )
    
    # Define start and end points (lon, lat)
    start_point = (12.315, 45.433)  # Example: near Venice
    end_point = (12.515, 45.233)    # Example: offshore
    
    # Extract transect with 50 points
    ds_transect = extractor.extract_transect(
        start_point=start_point,
        end_point=end_point,
        num_points=50,
        sort_direction="longitude",
        save_frequency=1
    )
    
    print(f"Transect extracted with {len(ds_transect.node)} points")
    
    # Calculate along-transect distance
    if 'distance' in ds_transect:
        print(f"Transect length: {ds_transect.distance.values[-1]:.0f} meters")
    
    extractor.close()
    return ds_transect


# Example 4: Batch processing multiple rivers
def example_batch_processing():
    """Process multiple river branches in one go."""
    
    simulation = "z025_barr003"
    nc_file = f"/path/to/simulations/{simulation}/output.nc"
    
    river_branches = [
        ("PoDiVenezia", "/path/to/nodes_PoDiVenezia.csv", "longitude"),
        ("PoDiGoro", "/path/to/nodes_PoDiGoro.csv", "longitude"),
        ("PoDiGnocca", "/path/to/nodes_PoDiGnocca.csv", "longitude"),
        ("PoDiTolle", "/path/to/nodes_PoDiTolle.csv", "latitude"),  # North-south river
    ]
    
    output_files = []
    
    for river_name, river_file, sort_dir in river_branches:
        print(f"Processing {river_name}...")
        
        extractor = SHYFEMNodeExtractor(nc_file, river_file)
        
        try:
            output_file = extractor.extract_nodes(
                output_prefix=f"{simulation}_{river_name}",
                sort_direction=sort_dir,
                save_frequency=1
            )
            output_files.append(output_file)
            print(f"  -> {output_file}")
            
        finally:
            extractor.close()
    
    return output_files


# Example 5: Interactive exploration
def example_interactive():
    """Interactive example with plotting."""
    
    extractor = SHYFEMNodeExtractor(
        nc_file="/path/to/tracer_file.nc"
    )
    
    # Load dataset to explore
    extractor._load_dataset()
    ds = extractor.ds
    
    print("Available variables:")
    for var in ds.data_vars:
        print(f"  - {var}: {ds[var].attrs.get('long_name', '')}")
    
    # Find nodes near a specific point
    point_of_interest = (12.35, 45.44)
    node_info = extractor.find_nearest_nodes([point_of_interest], return_indices=False)
    
    print(f"\nNearest node to {point_of_interest}:")
    print(f"  Coordinates: ({node_info.longitude[0]:.4f}, {node_info.latitude[0]:.4f})")
    print(f"  Node index: {node_info.node_index[0]}")
    
    # Extract time series at this point
    if 'salinity' in ds:
        salinity_ts = ds.salinity.isel(node=node_info.node_index[0])
        
        # Plot time series
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Surface salinity
        salinity_ts.isel(level=0).plot(ax=axes[0])
        axes[0].set_title("Surface salinity at point")
        
        # Bottom salinity
        salinity_ts.isel(level=-1).plot(ax=axes[1], color='red')
        axes[1].set_title("Bottom salinity at point")
        
        plt.tight_layout()
        plt.savefig("point_timeseries.png", dpi=150)
    
    extractor.close()


if __name__ == "__main__":
    # Run examples
    print("Example 0: Transect extraction from shapefile to netCDF")
    ds = example_transect_from_shapefile()

    print("Example 1: Basic extraction")
    ds1 = example_basic()
    
    print("\nExample 2: Convenience function")
    output_file = example_convenience()
    
    print("\nExample 3: Arbitrary transect")
    ds3 = example_arbitrary_transect()
    
    print("\nExample 4: Batch processing")
    files = example_batch_processing()
    
    print("\nAll examples completed!")