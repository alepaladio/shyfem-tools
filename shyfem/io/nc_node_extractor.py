"""
Module for extracting specific nodes from SHYFEM NetCDF files using xarray
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
from geopy.distance import geodesic
from typing import Optional, List, Union, Dict, Tuple
import warnings


class SHYFEMNodeExtractor:
    """
    Extractor class for selecting nodes from SHYFEM NetCDF files.
    
    Supports both hydro (water levels, velocities) and tracer 
    (salinity, temperature) files.
    """
    
    # Common coordinate name mappings
    COORD_MAPPINGS = {
        'longitude': ['Mesh2_node_x', 'lon', 'longitude', 'x'],
        'latitude': ['Mesh2_node_y', 'lat', 'latitude', 'y'],
        'depth': ['level', 'depth', 'nav_lev', 'z'],
        'time': ['time', 't', 'time_counter']
    }
    
    def __init__(self, nc_file: str, river_file: Optional[str] = None, 
                 output_dir: Optional[str] = None):
        """
        Initialize the extractor with file paths.
        
        Parameters
        ----------
        nc_file : str
            Path to input NetCDF file
        river_file : str, optional
            Path to CSV with river coordinates (required for extract_nodes)
        output_dir : str, optional
            Directory for output files. Defaults to input file directory.
        """
        self.nc_file = nc_file
        
        if river_file is not None:
            self.river_file = river_file
            self.river_data = None  # Will be loaded lazily
        else:
            self.river_data = None
        
        if output_dir is None:
            self.output_dir = os.path.dirname(nc_file)
        else:
            self.output_dir = output_dir
        
        # Load dataset to check content (lazy loading)
        self.ds = None  # Will load lazily when needed
        self._file_type = None  # 'hydro', 'tracer', or 'mixed'
        
    def _load_dataset(self):
        """Load xarray dataset if not already loaded."""
        if self.ds is None:
            try:
                self.ds = xr.open_dataset(self.nc_file)
                
                # Standardize coordinate names
                self._standardize_coords()
                
                # Determine file type based on available variables
                self._determine_file_type()
                
            except Exception as e:
                raise ValueError(f"Error loading NetCDF file: {e}")
    
    def _standardize_coords(self):
        """Standardize coordinate names using known mappings."""
        for standard_name, possible_names in self.COORD_MAPPINGS.items():
            for name in possible_names:
                if name in self.ds.coords or name in self.ds.dims:
                    if name != standard_name and standard_name not in self.ds.coords:
                        self.ds = self.ds.rename({name: standard_name})
                    break
    
    def _determine_file_type(self):
        """Determine if file contains hydro, tracer, or both data."""
        has_hydro = any(var in self.ds for var in ['water_level', 'zeta', 'u_velocity', 'v_velocity'])
        has_tracer = any(var in self.ds for var in ['salinity', 'temperature', 'salt', 'temp'])
        
        if has_hydro and has_tracer:
            self._file_type = 'mixed'
        elif has_hydro:
            self._file_type = 'hydro'
        elif has_tracer:
            self._file_type = 'tracer'
        else:
            warnings.warn("Could not determine file type from variables")
            self._file_type = 'unknown'
    
    def _load_river_data(self) -> pd.DataFrame:
        """Load river node data from CSV file."""
        if self.river_data is None:
            if not hasattr(self, 'river_file') or self.river_file is None:
                raise ValueError("No river file provided")
            self.river_data = pd.read_csv(self.river_file)
            
            # Standardize column names
            column_mapping = {}
            for col in self.river_data.columns:
                col_lower = col.lower()
                if 'node' in col_lower or 'id' in col_lower:
                    column_mapping[col] = 'node_id'
                elif 'lat' in col_lower:
                    column_mapping[col] = 'latitude'
                elif 'lon' in col_lower:
                    column_mapping[col] = 'longitude'
            
            if column_mapping:
                self.river_data = self.river_data.rename(columns=column_mapping)
        
        return self.river_data
    
    def _sort_river_nodes(self, river_df: pd.DataFrame, 
                         sort_direction: str = 'longitude') -> pd.DataFrame:
        """
        Sort river nodes along a specified direction and calculate distances.
        
        Parameters
        ----------
        river_df : pd.DataFrame
            DataFrame with at least 'latitude' and 'longitude' columns
        sort_direction : str
            'longitude' for west-east, 'latitude' for north-south
        
        Returns
        -------
        pd.DataFrame
            Sorted DataFrame with 'distance' column (cumulative distance in meters)
        """
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude']
        if not all(col in river_df.columns for col in required_cols):
            raise ValueError(f"River data must contain: {required_cols}")
        
        # Sort by specified direction
        if sort_direction not in ['longitude', 'latitude']:
            raise ValueError("sort_direction must be 'longitude' or 'latitude'")
        
        sorted_df = river_df.sort_values(by=sort_direction).reset_index(drop=True)
        
        # Calculate cumulative distance along the river
        latitudes = sorted_df['latitude'].values
        longitudes = sorted_df['longitude'].values
        
        distances = [0.0]  # Start with 0 for first point
        for i in range(len(latitudes) - 1):
            p1 = (latitudes[i], longitudes[i])
            p2 = (latitudes[i + 1], longitudes[i + 1])
            dist = geodesic(p1, p2).meters
            distances.append(dist)
        
        # Add cumulative distance
        sorted_df = sorted_df.copy()
        sorted_df['distance'] = np.cumsum(distances)
        
        return sorted_df
    
    def find_nearest_nodes(self, points: Union[List[Tuple[float, float]], pd.DataFrame], 
                          return_indices: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """
        Find nearest model nodes to given points.
        
        Parameters
        ----------
        points : list of (lon, lat) tuples or DataFrame with 'longitude', 'latitude' columns
            Points to find nearest nodes for
        return_indices : bool
            If True, return node indices, else return node coordinates
        
        Returns
        -------
        np.ndarray or pd.DataFrame
            Node indices or coordinates
        """
        self._load_dataset()
        
        # Convert input to standardized format
        if isinstance(points, pd.DataFrame):
            if not all(col in points.columns for col in ['longitude', 'latitude']):
                raise ValueError("DataFrame must have 'longitude' and 'latitude' columns")
            lons = points['longitude'].values
            lats = points['latitude'].values
        else:
            lons = np.array([p[0] for p in points])
            lats = np.array([p[1] for p in points])
        
        # Get model coordinates
        model_lons = self.ds['Mesh2_node_x'].values
        model_lats = self.ds['Mesh2_node_y'].values
        
        # Find nearest nodes for each point
        node_indices = []
        for lon, lat in zip(lons, lats):
            # Calculate Euclidean distance (fast approximation for small distances)
            distances = np.sqrt((model_lons - lon)**2 + (model_lats - lat)**2)
            nearest_idx = np.argmin(distances)
            node_indices.append(nearest_idx)
        
        node_indices = np.array(node_indices)
        
        if return_indices:
            return node_indices
        else:
            # Return coordinates of nearest nodes
            return pd.DataFrame({
                'longitude': model_lons[node_indices],
                'latitude': model_lats[node_indices],
                'node_index': node_indices
            })
    
    def extract_nodes(self, output_prefix: Optional[str] = None,
                     sort_direction: str = 'longitude',
                     time_slice: Optional[slice] = None,
                     save_frequency: int = 1,
                     variables: Optional[List[str]] = None) -> str:
        """
        Extract nodes along a river and create subset NetCDF file.
        
        Parameters
        ----------
        output_prefix : str, optional
            Prefix for output filename
        sort_direction : str
            Sorting direction: 'longitude' (west-east) or 'latitude' (north-south)
        time_slice : slice, optional
            Slice object for time selection (e.g., slice(0, 100))
        save_frequency : int
            Save every nth time step
        variables : list of str, optional
            Specific variables to extract. If None, extract all data variables.
        
        Returns
        -------
        str
            Path to created output file
        """
        # Load river data
        river_df = self._load_river_data()
        
        # Sort river nodes
        sorted_river = self._sort_river_nodes(river_df, sort_direction)
        
        # Find nearest model nodes
        node_indices = self.find_nearest_nodes(sorted_river)
        
        # Load dataset
        self._load_dataset()
        
        # Select time steps if specified
        if time_slice is not None:
            ds_subset = self.ds.isel(time=time_slice)
        else:
            ds_subset = self.ds
        
        # Apply save frequency to time dimension
        if 'time' in ds_subset.dims and save_frequency > 1:
            ds_subset = ds_subset.isel(time=slice(None, None, save_frequency))
        
        # Select specific nodes
        # ds_subset = ds_subset.isel(node=node_indices)
        ds_subset = ds_subset.isel(nMesh2_node=node_indices)
        
        # Select specific variables if requested
        if variables is not None:
            # Always keep coordinates
            coord_vars = list(ds_subset.coords)
            ds_subset = ds_subset[variables + coord_vars]
        
        # Add river metadata as attributes
        ds_subset.attrs['river_nodes_original'] = str(river_df.to_dict())
        ds_subset.attrs['river_nodes_sorted'] = str(sorted_river.to_dict())
        ds_subset.attrs['sort_direction'] = sort_direction
        ds_subset.attrs['node_indices'] = str(node_indices.tolist())
        
        # Determine output filename
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(self.nc_file))[0]
            river_name = os.path.splitext(os.path.basename(self.river_file))[0]
            output_prefix = f"{base_name}_{river_name}"
        
        output_file = os.path.join(
            self.output_dir,
            f"{output_prefix}_extracted.nc"
        )
        
        # Save to NetCDF
        encoding = self._get_encoding(ds_subset)
        ds_subset.to_netcdf(output_file, encoding=encoding)
        
        print(f"Extracted {len(node_indices)} nodes to: {output_file}")
        return output_file
    
    def _get_encoding(self, ds: xr.Dataset) -> Dict:
        """
        Get encoding settings for NetCDF export.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to encode
        
        Returns
        -------
        Dict
            Encoding settings
        """
        encoding = {}
        
        for var_name in ds.variables:
            var = ds[var_name]
            
            # Set appropriate data types
            if var_name in ['longitude', 'latitude', 'time']:
                encoding[var_name] = {'dtype': 'float64'}
            elif var_name in ['water_level', 'u_velocity', 'v_velocity', 
                            'salinity', 'temperature']:
                encoding[var_name] = {
                    'dtype': 'float32',
                    '_FillValue': None,
                    'zlib': True,
                    'complevel': 1
                }
            elif var.dtype in [np.float32, np.float64]:
                encoding[var_name] = {'dtype': 'float32', '_FillValue': None}
            else:
                encoding[var_name] = {'_FillValue': None}
        
        return encoding
    
    def extract_transect(self, start_point: Tuple[float, float], 
                        end_point: Tuple[float, float],
                        num_points: int = 100,
                        **kwargs) -> xr.Dataset:
        """
        Extract a transect between two points by interpolating nodes.
        
        Parameters
        ----------
        start_point : tuple (lon, lat)
            Starting point of transect
        end_point : tuple (lon, lat)
            Ending point of transect
        num_points : int
            Number of points along transect
        **kwargs
            Additional arguments passed to extract_nodes
        
        Returns
        -------
        xr.Dataset
            Dataset with interpolated transect
        """
        # Create evenly spaced points along the transect
        start_lon, start_lat = start_point
        end_lon, end_lat = end_point
        
        lons = np.linspace(start_lon, end_lon, num_points)
        lats = np.linspace(start_lat, end_lat, num_points)
        
        transect_points = pd.DataFrame({
            'longitude': lons,
            'latitude': lats
        })
        
        # Find nearest nodes for each transect point
        node_indices = self.find_nearest_nodes(transect_points)
        
        # Create a temporary river file for extraction
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            transect_points.to_csv(f.name, index=False)
            temp_river_file = f.name
        
        # Create new extractor instance for this transect
        extractor = SHYFEMNodeExtractor(self.nc_file, temp_river_file, self.output_dir)
        
        try:
            # Extract nodes
            output_file = extractor.extract_nodes(**kwargs)
            
            # Load and return the dataset
            ds_transect = xr.open_dataset(output_file)
            
            # Add transect metadata
            ds_transect.attrs['transect_start'] = str(start_point)
            ds_transect.attrs['transect_end'] = str(end_point)
            ds_transect.attrs['num_points'] = num_points
            
            return ds_transect
            
        finally:
            # Clean up temporary file
            os.unlink(temp_river_file)
    
    def close(self):
        """Close the dataset if it's open."""
        if self.ds is not None:
            self.ds.close()
            self.ds = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for quick extraction
def extract_river_transect(nc_file: str, river_file: str, 
                          output_file: Optional[str] = None,
                          **kwargs) -> str:
    """
    Convenience function for extracting river transects.
    
    Parameters
    ----------
    nc_file : str
        Path to NetCDF file
    river_file : str
        Path to river CSV file
    output_file : str, optional
        Output file path. If None, auto-generated.
    **kwargs
        Additional arguments for SHYFEMNodeExtractor.extract_nodes
    
    Returns
    -------
    str
        Path to output file
    """
    if output_file is None:
        output_dir = os.path.dirname(nc_file)
        base_name = os.path.splitext(os.path.basename(nc_file))[0]
        river_name = os.path.splitext(os.path.basename(river_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_{river_name}.nc")
    
    extractor = SHYFEMNodeExtractor(nc_file, river_file)
    
    try:
        return extractor.extract_nodes(output_prefix=os.path.splitext(
            os.path.basename(output_file))[0], **kwargs)
    finally:
        extractor.close()