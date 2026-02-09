"""
Module for extracting specific nodes from SHYFEM NetCDF files
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from geopy.distance import geodesic

class SHYFEMNodeExtractor:
    def __init__(self, nc_file, river_file, output_dir=None):
        """
        Initialize the extractor with file paths
        
        Args:
            nc_file (str): Path to input NetCDF file
            river_file (str): Path to CSV with river coordinates
            output_dir (str, optional): Directory for output files. Defaults to input file directory.
        """
        self.nc_file = nc_file
        self.river_file = river_file
        
        if output_dir is None:
            self.output_dir = os.path.dirname(nc_file)
        else:
            self.output_dir = output_dir
        
        # Determine file type (hydro or tracers) based on variables
        with nc.Dataset(nc_file, 'r') as ds:
            self.is_hydro = 'water_level' in ds.variables
            self.is_tracer = 'salinity' in ds.variables
        
        if not (self.is_hydro or self.is_tracer):
            raise ValueError("Input file doesn't contain recognized hydro or tracer variables")
    
    def _sort_river_nodes(self, sort_LonLat=0):
        """
        Internal method to sort river nodes
        
        Args:
            sort_LonLat (int): 0 for west-east, 1 for north-south sorting
            
        Returns:
            pd.DataFrame: Sorted nodes with columns Node, Lat, Lon, Distance
        """
        river_df = pd.read_csv(self.river_file)
        
        # Determine sort column
        sort_col = 'Lon' if sort_LonLat == 0 else 'Lat'
        river_df = river_df.sort_values(by=sort_col)
        
        # Calculate distances between nodes
        lats = river_df['Lat'].values
        lons = river_df['Lon'].values
        nodes = river_df['Node'].values
        
        distances = [0]
        for i in range(len(nodes) - 1):
            p1 = (lats[i], lons[i])
            p2 = (lats[i+1], lons[i+1])
            dist = geodesic(p1, p2).meters
            
            # Handle large gaps between nodes
            if dist > 50:
                # Find closest next node within 100m
                for j in range(i+1, len(nodes)):
                    pj = (lats[j], lons[j])
                    new_dist = geodesic(p1, pj).meters
                    if new_dist < 100:
                        # Swap nodes
                        nodes[i+1], nodes[j] = nodes[j], nodes[i+1]
                        lats[i+1], lats[j] = lats[j], lats[i+1]
                        lons[i+1], lons[j] = lons[j], lons[i+1]
                        break
                else:
                    # If no close node found, find the closest one
                    dists = [geodesic(p1, (lats[k], lons[k])).meters 
                            for k in range(i+1, len(nodes))]
                    min_idx = np.argmin(dists) + i + 1
                    nodes[i+1], nodes[min_idx] = nodes[min_idx], nodes[i+1]
                    lats[i+1], lats[min_idx] = lats[min_idx], lats[i+1]
                    lons[i+1], lons[min_idx] = lons[min_idx], lons[i+1]
            
            distances.append(geodesic(p1, (lats[i+1], lons[i+1])).meters)
        
        return pd.DataFrame({
            'Node': nodes,
            'Lat': lats,
            'Lon': lons,
            'Distance': distances
        })
    
    def extract_nodes(self, output_prefix=None, sort_LonLat=0, 
                     time_steps=None, save_frequency=1):
        """
        Extract nodes and create subset NetCDF file
        
        Args:
            output_prefix (str, optional): Prefix for output filename
            sort_LonLat (int): Sorting direction (0=west-east, 1=north-south)
            time_steps (list, optional): Specific time steps to extract
            save_frequency (int): Save every nth time step
            
        Returns:
            str: Path to created output file
        """
        # Process river nodes
        river_nodes = self._sort_river_nodes(sort_LonLat)
        
        # Get base filename for output
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(self.nc_file))[0]
            output_prefix = f"{base_name}_extracted"
        
        output_file = os.path.join(
            self.output_dir,
            f"{output_prefix}.nc"
        )
        
        # Determine which variables to process
        with nc.Dataset(self.nc_file, 'r') as src:
            # Get common variables
            lon = src.variables['Mesh2_node_x'][:]
            lat = src.variables['Mesh2_node_y'][:]
            levels = src.variables['level'][:]
            depths = src.variables['total_depth'][:]
            time = src.variables['time'][:]
            
            # Find nearest nodes in original grid
            node_indices = []
            for r_lon, r_lat in zip(river_nodes['Lon'], river_nodes['Lat']):
                dist = np.sqrt((lon - r_lon)**2 + (lat - r_lat)**2)
                node_indices.append(np.argmin(dist))
            
            # Determine time steps to save
            if time_steps is None:
                time_steps = np.arange(0, len(time), save_frequency)
            else:
                time_steps = np.array(time_steps)[::save_frequency]
            
            # Create output file
            with nc.Dataset(output_file, 'w', format='NETCDF4') as dst:
                # Create dimensions
                dst.createDimension('node', len(node_indices))
                dst.createDimension('level', len(levels))
                dst.createDimension('time', len(time_steps))
                
                # Copy common variables
                for name, var in src.variables.items():
                    if name in ['Mesh2_node_x', 'Mesh2_node_y', 'level', 
                              'total_depth', 'time']:
                        new_var = dst.createVariable(
                            name, var.datatype, var.dimensions
                        )
                        new_var[:] = var[:]
                        for attr in var.ncattrs():
                            new_var.setncattr(attr, var.getncattr(attr))
                
                # Process hydro variables if present
                if self.is_hydro:
                    for var_name in ['water_level', 'u_velocity', 'v_velocity']:
                        if var_name in src.variables:
                            src_var = src.variables[var_name]
                            dims = ('time', 'node') if var_name == 'water_level' else ('time', 'node', 'level')
                            dst_var = dst.createVariable(
                                var_name, src_var.datatype, dims
                            )
                            
                            for attr in src_var.ncattrs():
                                dst_var.setncattr(attr, src_var.getncattr(attr))
                            
                            # Copy data
                            for i, t in enumerate(time_steps):
                                if var_name == 'water_level':
                                    dst_var[i, :] = src_var[t, node_indices]
                                else:
                                    dst_var[i, :, :] = src_var[t, node_indices, :]
                
                # Process tracer variables if present
                if self.is_tracer:
                    for var_name in ['salinity', 'temperature']:
                        if var_name in src.variables:
                            src_var = src.variables[var_name]
                            dst_var = dst.createVariable(
                                var_name, src_var.datatype, ('time', 'node', 'level')
                            )
                            
                            for attr in src_var.ncattrs():
                                dst_var.setncattr(attr, src_var.getncattr(attr))
                            
                            # Copy data
                            for i, t in enumerate(time_steps):
                                dst_var[i, :, :] = src_var[t, node_indices, :]
        
        return output_file
