"""
Example usage of SHYFEMNodeExtractor with xarray
"""

from shyfem.io.nc_node_extractor import SHYFEMNodeExtractor, extract_river_transect
import xarray as xr
import matplotlib.pyplot as plt

# Example 1: Basic extraction
def example_basic():
    """Basic example of extracting river nodes."""
    
    # Initialize extractor
    extractor = SHYFEMNodeExtractor(
        nc_file="/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/deltapo_ER_202207_ogridNoDRCVmin_WATEXT_ts_uns.nc",
        river_file="/home/utente/Documenti/climaxpo/qgis/river_branches/nodes_PoDiVeneziaF.csv",
        output_dir="/home/utente/Documenti/climaxpo/climaxpo_2022-2023/test_results"
    )
    
    # Extract nodes along river (west-east sorting)
    output_file = extractor.extract_nodes(
        output_prefix="Po_di_Venezia",
        sort_direction="longitude",  # or "latitude"
        save_frequency=2,  # Save every 2nd time step
        variables=['salinity', 'temperature']  # Optional: select specific variables
    )
    
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
    print("Example 1: Basic extraction")
    # ds1 = example_basic()
    
    print("\nExample 2: Convenience function")
    output_file = example_convenience()
    
    print("\nExample 3: Arbitrary transect")
    ds3 = example_arbitrary_transect()
    
    print("\nExample 4: Batch processing")
    files = example_batch_processing()
    
    print("\nAll examples completed!")