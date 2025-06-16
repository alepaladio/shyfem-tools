from shyfem.io.nc_node_extractor import SHYFEMNodeExtractor

# Example for hydro file
hydro_extractor = SHYFEMNodeExtractor(
    nc_file="/path/to/hydro_file.nc",
    river_file="/path/to/river_nodes.csv",
    output_dir="/path/to/output"
)

hydro_output = hydro_extractor.extract_nodes(
    output_prefix="PoDiVenezia_hydro",
    sort_LonLat=0,
    save_frequency=2
)

# Example for tracer file
tracer_extractor = SHYFEMNodeExtractor(
    nc_file="/path/to/tracer_file.nc",
    river_file="/path/to/river_nodes.csv",
    output_dir="/path/to/output"
)

tracer_output = tracer_extractor.extract_nodes(
    output_prefix="PoDiVenezia_tracer",
    sort_LonLat=0,
    time_steps=range(0, 100)  # Only first 100 time steps
)
