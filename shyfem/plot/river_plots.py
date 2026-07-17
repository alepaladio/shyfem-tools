# -*- coding: utf-8 -*-
"""
River transect plotting functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as tri
from netCDF4 import Dataset, num2date
from datetime import datetime
from typing import Optional, Tuple, List, Union, Dict
from dataclasses import dataclass, field
import cmocean.cm as cmo

from ..io.nc_node_extractor import SHYFEMNodeExtractor
from .utils import haversine, save_figure, create_video


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class RiverPlotConfig:
    """Configuration for river transect plots."""
    # Data flags
    plot_hydro: bool = False
    plot_ts: bool = True
    plot_measurements: bool = False
    make_video: bool = True
    
    # Plot limits
    x_lim: Tuple[float, float] = (0, 50000)
    y_lim: Tuple[float, float] = (-9.5, 1)
    max_discharge_lim: float = 750
    
    # Time settings
    time_units: str = 'hours since 2022-07-15 00:00:00'
    time_calendar: str = 'standard'
    date_range: Optional[Tuple[str, str]] = None
    
    # Contour levels
    salinity_levels: np.ndarray = field(
        default_factory=lambda: np.array([0, 2, 5, 10, 15, 20, 25, 30, 35, 40])
    )
    temperature_levels: np.ndarray = field(
        default_factory=lambda: np.array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    )
    
    # Output settings
    output_folder: Optional[str] = None
    fps: int = 4
    dpi: int = 100
    
    def __post_init__(self):
        pass


# ============================================================================
# MAIN PLOTTER CLASS
# ============================================================================

class RiverTransectPlotter:
    """Plotter for river transect data."""
    
    # ------------------------------------------------------------------------
    # INITIALIZATION AND DATA LOADING
    # ------------------------------------------------------------------------
    
    def __init__(self, config: RiverPlotConfig):
        self.config = config
        self.hydro_data = None
        self.ts_data = None
        self.discharge_dates = None
        self.discharge_values = None
        self.measurement_data = None
        self.store_swi = []
        self.image_paths = []
    
    def load_discharge_data(self, discharge_file: str):
        dates, values = self._read_dat_file(discharge_file)
        self.discharge_dates = dates
        self.discharge_values = values
    
    def load_measurement_data(self, measurement_files: dict, dist_points: list):
        self.measurement_data = {}
        for file_id, dist_point in zip(measurement_files.keys(), dist_points):
            self.measurement_data[file_id] = self._read_measurement_file(measurement_files[file_id])
    
    def load_model_data(self, models_folder: str, river_branch: str):
        if self.config.plot_hydro:
            hydro_file = f"{models_folder}/{river_branch}_hydro_extracted.nc"
            self.hydro_data = self._load_hydro_netcdf(hydro_file)
        
        if self.config.plot_ts:
            ts_file = f"{models_folder}/{river_branch}_ts_extracted.nc"
            self.ts_data = self._load_ts_netcdf(ts_file)
    
    # ------------------------------------------------------------------------
    # INTERNAL FILE READING METHODS
    # ------------------------------------------------------------------------
    
    def _read_dat_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        dates, values = [], []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                date_str, value = parts[0], float(parts[1])
                date = datetime.strptime(date_str, '%Y-%m-%d::%H:%M:%S')
                dates.append(date)
                values.append(value)
        return np.array(dates), np.array(values)
    
    def _read_measurement_file(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(filename, sep='\t', parse_dates=['Date'],
                          date_format='%Y-%m-%d::%H:%M:%S')
    
    def _load_hydro_netcdf(self, filename: str):
        ds = Dataset(filename)
        t = num2date(ds.variables['time'][:], units=self.config.time_units, 
                     calendar=self.config.time_calendar)
        level = ds.variables['depth'][:]
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        td = ds.variables['total_depth'][:] * -1
        wl = ds.variables['water_level'][:]
        uv = ds.variables['u_velocity'][:]
        vv = ds.variables['v_velocity'][:]
        
        dist = self._calculate_distances(lat, lon)
        X, Y = np.meshgrid(dist, level * -1)
        
        wl_dates = np.array([datetime(t_i.year, t_i.month, t_i.day, 
                                       t_i.hour, t_i.minute, t_i.second) for t_i in t])
        wl_values = wl[:, -1]
        ds.close()
        
        return {
            'time': t, 'level': level, 'lon': lon, 'lat': lat,
            'total_depth': td, 'water_level': wl, 'u_velocity': uv,
            'v_velocity': vv, 'wl_dates': wl_dates, 'wl_values': wl_values,
            'X': X, 'Y': Y, 'distance': dist
        }
    
    def _load_ts_netcdf(self, filename: str):
        ds = Dataset(filename)
        sal = ds.variables['salinity'][:]
        tem = ds.variables['temperature'][:]
        t = num2date(ds.variables['time'][:], units=self.config.time_units,
                     calendar=self.config.time_calendar)
        level = ds.variables['depth'][:]
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        td = ds.variables['total_depth'][:] * -1
        
        dist = self._calculate_distances(lat, lon)
        dist = np.flipud(dist)
        X, Y = np.meshgrid(dist, level * -1)
        
        ts_dates = np.array([datetime(t_i.year, t_i.month, t_i.day,
                                       t_i.hour, t_i.minute, t_i.second) for t_i in t])
        ds.close()
        
        return {
            'time': t, 'level': level, 'lon': lon, 'lat': lat,
            'total_depth': td, 'salinity': sal, 'temperature': tem,
            'ts_dates': ts_dates, 'X': X, 'Y': Y, 'distance': dist
        }
    
    def _load_ts_netcdf_full(self, filename: str):
        """Load full TS NetCDF data."""
        ds = Dataset(filename)
        
        sal = ds.variables['salinity'][:]
        tem = ds.variables['temperature'][:]
        
        # Read time units from the NetCDF file
        time_units = ds.variables['time'].units
        time_calendar = getattr(ds.variables['time'], 'calendar', 'standard')
        t = num2date(ds.variables['time'][:], units=time_units, calendar=time_calendar)
        
        level = ds.variables['level'][:]
        lon = ds.variables['Mesh2_node_x'][:]
        lat = ds.variables['Mesh2_node_y'][:]
        td = ds.variables['total_depth'][:] * -1
        
        ds.close()
        
        return {
            'time': t,
            'level': level,
            'lon': lon,
            'lat': lat,
            'total_depth': td,
            'salinity': sal,
            'temperature': tem,
            'time_units': time_units,
            'time_calendar': time_calendar
        }
    
    def _load_hydro_netcdf_full(self, filename: str):
        """Load full Hydro NetCDF data."""
        ds = Dataset(filename)
        
        # Read time units from the NetCDF file
        time_units = ds.variables['time'].units
        time_calendar = getattr(ds.variables['time'], 'calendar', 'standard')
        t = num2date(ds.variables['time'][:], units=time_units, calendar=time_calendar)
        
        level = ds.variables['level'][:]
        lon = ds.variables['Mesh2_node_x'][:]
        lat = ds.variables['Mesh2_node_y'][:]
        td = ds.variables['total_depth'][:] * -1
        wl = ds.variables['water_level'][:]
        uv = ds.variables['u_velocity'][:]
        vv = ds.variables['v_velocity'][:]
        
        ds.close()
        
        return {
            'time': t,
            'level': level,
            'lon': lon,
            'lat': lat,
            'total_depth': td,
            'water_level': wl,
            'u_velocity': uv,
            'v_velocity': vv,
            'time_units': time_units,
            'time_calendar': time_calendar
        }
    
    def read_grd(self, filename):
        """Read GRD file and extract points and elements."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            points = {}
            elements = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Parse points (type 1)
                if len(parts) >= 5 and parts[0] == '1':
                    point_id = int(parts[1])
                    point_type = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5]) if len(parts) >= 6 else 0.0
                    points[point_id] = {'x': x, 'y': y, 'type': point_type, 'depth': z}
                
                # Parse elements (type 2)
                elif len(parts) >= 7 and parts[0] == '2':
                    element_id = int(parts[1])
                    element_type = int(parts[2])
                    num_points = int(parts[3])
                    node_ids = [int(parts[4]), int(parts[5]), int(parts[6])]
                    depth = float(parts[7]) if len(parts) > 7 else 0.0
                    elements.append({
                        'id': element_id,
                        'type': element_type,
                        'num_points': num_points,
                        'node_ids': node_ids,
                        'depth': depth
                    })
            
            print(f"Loaded GRD: {len(points)} points, {len(elements)} elements")
            return points, elements
            
        except Exception as e:
            print(f"Error reading GRD: {e}")
            return None, None
    
    def _calculate_distances(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        c_d = [haversine(lat[i], lon[i], lat[i + 1], lon[i + 1]) 
               for i in range(len(lat) - 1)]
        c_dd = [i * 1000 for i in c_d]
        c_d = [0] + c_dd
        return np.cumsum(c_d)
    
    # ------------------------------------------------------------------------
    # QUIVER HELPER FUNCTIONS
    # ------------------------------------------------------------------------
    
    def _add_quivers(self, ax, c_uv, c_vv, X, Y1, n=3, m=10):
        magnitude = np.sqrt(c_uv**2 + c_vv**2)
        un = c_uv / magnitude
        vn = c_vv / magnitude
        un = np.nan_to_num(un)
        vn = np.nan_to_num(vn)
        
        u_direction = np.sign(c_uv[0::n, 0::m])
        v_direction = np.zeros_like(u_direction)
        
        mask_right = u_direction > 0
        mask_left = u_direction < 0
        X_flipped = np.fliplr(X)
        
        ax.quiver(X_flipped[0::n, 0::m][mask_right], Y1[0::n, 0::m][mask_right],
                  u_direction[mask_right], v_direction[mask_right],
                  scale=6, scale_units='inches', width=0.002, headwidth=4,
                  color='black', label='Out', zorder=3)
        ax.quiver(X_flipped[0::n, 0::m][mask_left], Y1[0::n, 0::m][mask_left],
                  u_direction[mask_left], v_direction[mask_left],
                  scale=6, scale_units='inches', width=0.002, headwidth=4,
                  color='blue', label='In', zorder=3)
    
    def _add_quivers_cross_section(self, ax, c_uv, c_vv, X, Y, n=3, m=10):
        """Add quiver arrows to cross-section plot."""
        # Now do the bussiness 
        uv_sub = c_uv[::n, ::m]
        vv_sub = c_vv[::n, ::m]
        X_sub = X[::n, ::m]
        Y_sub = Y[::n, ::m]
        
        # Calculate magnitude
        magnitude = np.sqrt(uv_sub**2 + vv_sub**2)
        
        # Mask zero or very small velocities
        valid = magnitude > 0.01
        
        if not np.any(valid):
            return
        
        # Create mask for direction based on u_velocity
        mask_out = (uv_sub > 0) & valid
        mask_in = (uv_sub < 0) & valid
        
        # Plot outgoing arrows (seaward)
        if np.any(mask_out):
            ax.quiver(X_sub[mask_out], Y_sub[mask_out],
                      uv_sub[mask_out], vv_sub[mask_out],
                      scale=2, scale_units='inches', width=0.005, headwidth=4,
                      color='black', label='Out', zorder=3)
        
        # Plot incoming arrows (landward)
        if np.any(mask_in):
            ax.quiver(X_sub[mask_in], Y_sub[mask_in],
                      uv_sub[mask_in], vv_sub[mask_in],
                      scale=2, scale_units='inches', width=0.005, headwidth=4,
                      color='blue', label='In', zorder=3)
            
        print('done quivering')
    
    def _add_quiver_2DHmap(self, ax, triang, uv_data, vv_data, x_lims, y_lims, 
                            grid_resolution=5):
        """
        Add quiver arrows on a regular grid for 2D horizontal maps.
        """
        from scipy.spatial import cKDTree
        
        # Create regular grid
        x_grid = np.linspace(x_lims[0], x_lims[1], grid_resolution)
        y_grid = np.linspace(y_lims[0], y_lims[1], grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Flatten grid points
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Find nearest triangulation point for each grid point
        tri_points = np.column_stack([triang.x, triang.y])
        tree = cKDTree(tri_points)
        distances, indices = tree.query(grid_points)
        
        # Get velocity values at nearest points
        uv_grid = uv_data[indices].reshape(X_grid.shape)
        vv_grid = vv_data[indices].reshape(X_grid.shape)
        
        # Mask points that are too far from triangulation
        mask = distances.reshape(X_grid.shape) > 0.05
        uv_grid[mask] = np.nan
        vv_grid[mask] = np.nan
        
        # Calculate magnitude for masking
        mag = np.sqrt(uv_grid**2 + vv_grid**2)
        valid = mag > 0.01
        
        if not np.any(valid):
            return
        
        # Subsample grid for cleaner visualization
        subsample = 2
        X_sub = X_grid[::subsample, ::subsample]
        Y_sub = Y_grid[::subsample, ::subsample]
        uv_sub = uv_grid[::subsample, ::subsample]
        vv_sub = vv_grid[::subsample, ::subsample]
        valid_sub = valid[::subsample, ::subsample]
        
        # Create masks
        mask_out = (uv_sub > 0) & valid_sub
        mask_in = (uv_sub < 0) & valid_sub
        
        # Plot outgoing arrows (seaward)
        if np.any(mask_out):
            ax.quiver(X_sub[mask_out], Y_sub[mask_out],
                     uv_sub[mask_out], vv_sub[mask_out],
                     scale=2, scale_units='inches', width=0.002, headwidth=3,
                     color='black', label='Out', zorder=3)
        
        # Plot incoming arrows (landward)
        if np.any(mask_in):
            ax.quiver(X_sub[mask_in], Y_sub[mask_in],
                     uv_sub[mask_in], vv_sub[mask_in],
                     scale=2, scale_units='inches', width=0.002, headwidth=3,
                     color='blue', label='In', zorder=3)
    
    # ------------------------------------------------------------------------
    # MAP PANEL HELPER FUNCTIONS
    # ------------------------------------------------------------------------
    
    def _plot_map_panel(self, ax, triang, data, label, units, x_lims, y_lims,
                        levels, cmap, time_str, layer_name,
                        sal_limits=None, temp_limits=None, wl_limits=None, vel_limits=None,
                        has_hydro=False, uv_data=None, vv_data=None,
                        plot_quivers=False, quiver_grid_resolution=5, 
                        shapefiles=None, dat_files=None):
        """
        Helper function to plot a single map panel.
        """
        # Set limits
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        
        # Plot triangulation
        ax.triplot(triang, 'k-', lw=0.3, alpha=0.3)
        
        # Determine which limit to use based on label
        limits = None
        if 'Salinity' in label and sal_limits is not None:
            limits = sal_limits
        elif 'Temperature' in label and temp_limits is not None:
            limits = temp_limits
        elif 'Water Level' in label and wl_limits is not None:
            limits = wl_limits
        elif 'Velocity' in label and vel_limits is not None:
            limits = vel_limits
        
        # Use the limits in contourf
        if limits is not None:
            vmin, vmax = limits
            contour_plot = ax.tricontourf(triang, data, levels=levels, cmap=cmap, 
                                          vmin=vmin, vmax=vmax, extend='both')
        else:
            contour_plot = ax.tricontourf(triang, data, levels=levels, cmap=cmap, extend='both')
        
        ax.tricontour(triang, data, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour_plot, ax=ax, label=f'{label} ({units})', shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        
        # Add quivers if enabled and hydro data available
        if plot_quivers and has_hydro and uv_data is not None and vv_data is not None:
            self._add_quiver_2DHmap(ax, triang, uv_data, vv_data, x_lims, y_lims, 
                                    grid_resolution=quiver_grid_resolution)
        
        # Add shapefiles if provided
        if shapefiles:
            for shp_info in shapefiles:
                self._add_shapefile(ax, shp_info['path'], shp_info.get('type', 'auto'))
                
        # Add DAT lines if provided
        if dat_files:
            for dat_info in dat_files:
                if isinstance(dat_info, dict):
                    self._add_dat_line(ax, 
                                      dat_info['path'],
                                      color=dat_info.get('color', 'magenta'),
                                      linewidth=dat_info.get('linewidth', 2),
                                      alpha=dat_info.get('alpha', 0.8),
                                      label=dat_info.get('label', None))
                else:
                    # Simple string path
                    self._add_dat_line(ax, dat_info, color='magenta')
        
        # Add title
        ax.set_title(f'{label} - {layer_name} layer at {time_str}', fontsize=14)
        ax.grid(True, alpha=0.2)
    
    def _add_shapefile(self, ax, shapefile_path, shp_type='auto'):
        """Add shapefile overlay to axis."""
        import shapefile
        
        try:
            sf = shapefile.Reader(shapefile_path)
            shapes = sf.shapes()
            
            for shape in shapes:
                points = shape.points
                
                # Detect geometry type if auto
                if shp_type == 'auto':
                    if len(points) == 1:
                        shp_type = 'point'
                    elif len(points) > 1 and points[0] == points[-1]:
                        shp_type = 'polygon'
                    else:
                        shp_type = 'line'
                
                if shp_type == 'line':
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    ax.plot(x, y, 'g-', linewidth=2, alpha=0.8, zorder=4)
                
                elif shp_type == 'polygon':
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    ax.fill(x, y, 'none', edgecolor='g', linewidth=2, alpha=0.8, zorder=4)
                
                elif shp_type == 'point':
                    x = points[0][0]
                    y = points[0][1]
                    ax.scatter(x, y, c='g', s=50, marker='*', zorder=4)
            
        except Exception as e:
            print(f"Error adding shapefile {shapefile_path}: {e}")
            
    def _read_dat_nodes(self, filename: str) -> pd.DataFrame:
        """
        Read DAT file with node coordinates.
        Format: node,longitude,latitude[,depth]
        
        Parameters
        ----------
        filename : str
            Path to DAT file
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: node, longitude, latitude, depth (if available)
        """
        try:
            df = pd.read_csv(filename)
            
            # Check required columns
            required = ['node', 'longitude', 'latitude']
            if not all(col in df.columns for col in required):
                # Try alternative column names
                rename_map = {}
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if col_lower in ['node', 'node_id', 'point_id', 'id']:
                        rename_map[col] = 'node'
                    elif col_lower in ['lon', 'long', 'longitude', 'x']:
                        rename_map[col] = 'longitude'
                    elif col_lower in ['lat', 'latitude', 'y']:
                        rename_map[col] = 'latitude'
                    elif col_lower in ['depth', 'z', 'elevation']:
                        rename_map[col] = 'depth'
                
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                # Check again after rename
                if not all(col in df.columns for col in required):
                    print(f"Warning: DAT file {filename} missing required columns. "
                          f"Found: {list(df.columns)}")
                    return None
            
            return df
            
        except Exception as e:
            print(f"Error reading DAT file {filename}: {e}")
            return None
    
    def _add_dat_line(self, ax, dat_file, color='magenta', linewidth=2, 
                      alpha=0.8, label=None, zorder=5):
        """
        Add a line from DAT file to axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        dat_file : str
            Path to DAT file
        color : str
            Line color (default: 'magenta')
        linewidth : float
            Line width (default: 2)
        alpha : float
            Transparency (default: 0.8)
        label : str, optional
            Label for legend
        zorder : int
            Z-order for plotting (default: 5)
        """
        df = self._read_dat_nodes(dat_file)
        
        if df is None:
            return
        
        # Plot the line in the order of the file
        ax.plot(df['longitude'], df['latitude'], 
                color=color, linewidth=linewidth, alpha=alpha,
                label=label, zorder=zorder)
        
        # Optionally add markers at nodes
        ax.scatter(df['longitude'], df['latitude'], 
                  color=color, s=10, alpha=alpha*0.7, zorder=zorder+1)
        
    # ------------------------------------------------------------------------
    # SWI (Salt Wedge Indicator) HELPER FUNCTIONS
    # ------------------------------------------------------------------------
    
    def _detect_threshold_exceedance(self, c_sal, X, threshold=2):
        for col in range(c_sal.shape[1]):
            if np.any(c_sal[:, col] > threshold):
                return X[:, col], col
        return None, None
    
    def _add_mask(self, ax, data, X, Y1):
        mask = np.ones_like(data)
        mask[np.isnan(data)] = 0
        ax.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5],
                    alpha=1.0, zorder=1)
    
    def _add_swi_line(self, ax, X_length, col_X, c_t, Y1):
        x_value = X_length[0]
        ax.axvline(x=x_value, color='red', linestyle='--', linewidth=2,
                   label=f"Threshold @ {x_value:.2f}")
        x_value_km = x_value / 1000
        new_entry = [c_t, x_value_km,
                    np.flip(self.ts_data['lon'][col_X]),
                    np.flip(self.ts_data['lat'][col_X])]
        self.store_swi.append(new_entry)
        ax.text(x_value + 0.2, Y1.max(), f"{x_value_km:.2f} km", fontsize=12,
                color='black', bbox=dict(facecolor='white', edgecolor='black',
                                        boxstyle='round,pad=0.3'))
    
    def _add_max_swi_line(self, ax, Y1):
        store_swi_array = np.array(self.store_swi)
        max_swi_length = np.max(store_swi_array[:, 1])
        max_indices = np.where(store_swi_array[:, 1] == max_swi_length)[0]
        max_index = max_indices[-1]
        max_time, max_swi_length, max_lon, max_lat = store_swi_array[max_index]
        max_time_str = str(max_time)
        
        ax.axvline(x=max_swi_length * 1000, color='magenta', linestyle='--', linewidth=2,
                   label=f"Max SWI @ {max_swi_length:.2f} km")
        ax.text(max_swi_length * 1000 + 0.2, Y1.max() - 2,
                f"Max Length: {max_swi_length:.2f} km\n{max_time_str}",
                fontsize=12, color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    def _add_measurements(self, ax, c_t):
        for file_id, dist_point in zip(self.measurement_data.keys(), [48591, 48499, 46915, 44004, 41644, 38871, 35719]):
            data = self.measurement_data[file_id]
            c_dist = np.mean(data['Distance'])
            
            if self.config.plot_hydro:
                closest_idx = np.abs(self.hydro_data['distance'] - c_dist).argmin()
                closest_wl = self.hydro_data['water_level'][:, closest_idx]
            else:
                closest_idx = np.abs(self.ts_data['distance'] - c_dist).argmin()
                closest_wl = 0
            
            new_depth = pd.DataFrame({
                'Distance': [data['Distance'].iloc[0]],
                'Depth': [-closest_wl],
                'Salinity': [data['Salinity'].iloc[0]],
                'Date': [data['Date'].iloc[0]]
            })
            data = pd.concat([new_depth, data], ignore_index=True)
            
            ax.scatter(data['Distance'], data['Depth'] * -1, c=data['Salinity'],
                      cmap='turbo', vmin=1, vmax=35, edgecolors='w', s=10, zorder=5)
    
    # ------------------------------------------------------------------------
    # HOVMOLLER HELPER FUNCTIONS
    # ------------------------------------------------------------------------
    
    def get_bottom_salt_branch(self, t, c_branch_ts):
        """Get bottom layer salinity for each time step."""
        num_alongshore_points = c_branch_ts['salinity'].shape[1]
        num_time_steps = c_branch_ts['salinity'].shape[0]
        c_bot_salt = np.full((num_alongshore_points, num_time_steps), np.nan)
        
        for ii in range(len(t)):
            c_sal = np.transpose(c_branch_ts['salinity'][ii, :, :])
            c_sal[c_sal == 0] = np.nan
            mask = ~np.isnan(c_sal)
            last_valid_indices = np.argmax(mask[::-1, :], axis=0)
            last_valid_indices = c_sal.shape[0] - 1 - last_valid_indices
            c_bot_salt[:, ii] = c_sal[last_valid_indices, np.arange(c_sal.shape[1])]
            all_nan_columns = ~mask.any(axis=0)
            c_bot_salt[all_nan_columns, ii] = np.nan
        
        return c_bot_salt
    
    def get_surface_salt_branch(self, t, c_branch_ts):
        """Get surface layer salinity for each time step."""
        num_alongshore_points = c_branch_ts['salinity'].shape[1]
        num_time_steps = c_branch_ts['salinity'].shape[0]
        c_surf_salt = np.full((num_alongshore_points, num_time_steps), np.nan)
        
        for ii in range(num_time_steps):
            c_sal = np.transpose(c_branch_ts['salinity'][ii, :, :])
            c_sal[c_sal == 0] = np.nan
            first_valid_indices = np.argmax(~np.isnan(c_sal), axis=0)
            c_surf_salt[:, ii] = c_sal[first_valid_indices, np.arange(c_sal.shape[1])]
        
        return c_surf_salt
    
    # ------------------------------------------------------------------------
    # UTILITY FUNCTIONS
    # ------------------------------------------------------------------------
    
    def _get_suffix(self) -> str:
        if self.config.plot_hydro and self.config.plot_ts:
            return '_hydro_ts'
        elif self.config.plot_hydro:
            return '_hydro'
        elif self.config.plot_ts:
            return '_ts'
        return ''
    
    def _save_swi_data(self, output_folder: str, river_branch: str):
        f_store_swi = os.path.join(output_folder, f'{river_branch}_plots',
                                   f'results_{river_branch}_stored_swi.txt')
        with open(f_store_swi, 'w') as f:
            for row in self.store_swi:
                date_str = row[0].isoformat(sep='::')
                length_km = f"{row[1]:.4f}"
                lat = f"{row[2]:.8f}"
                lon = f"{row[3]:.8f}"
                f.write(f"{date_str}\t{length_km}\t{lat}\t{lon}\n")
    
    def _create_video(self, output_folder: str, river_branch: str):
        suffix = self._get_suffix()
        output_video = os.path.join(output_folder, f'{river_branch}_plots',
                                   f'{river_branch}{suffix}_video4.mp4')
        create_video(self.image_paths, output_video, fps=self.config.fps)
        print(f"Video saved at {output_video}")


# ============================================================================
# PLOT TYPE 1: 2D-V CROSS-SECTION
# ============================================================================

    def plot_cross_section(self, river_branch: str, extracted_folder: str, 
                           output_folder: Optional[str] = None,
                           nodes_file: Optional[str] = None,
                           time_idx: Union[int, List[int], str] = 0,
                           y_lims: Optional[List[float]] = None,
                           x_lims: Optional[List[float]] = None,
                           layer: str = 'all', 
                           save_plot: bool = True,
                           make_video: bool = True,
                           fps: int = 4,
                           quiver_n: int = 3,
                           quiver_m: int = 10):
        """
        Plot 2D vertical cross-section along the transect.
        x-axis = distance along transect, y-axis = depth.
        
        Parameters
        ----------
        river_branch : str
            Name of river branch
        extracted_folder : str
            Folder containing extracted NetCDF files
        output_folder : str, optional
            Output folder for plots (default: extracted_folder)
        nodes_file : str, optional
            Path to nodes file with depth data (node,longitude,latitude,depth)
        time_idx : int, list, or str
            Time index(es) to plot. Options:
            - int: single time step (e.g., 0)
            - list: range of time steps (e.g., [0, 9] or [5, 9])
            - str: 'all' for all time steps
        y_lims : list, optional
            Y-axis limits [min, max] (e.g., [-50, 0.5])
        x_lims : list, optional
            X-axis limits [min, max] (e.g., [0, 50000])
        layer : str
            'all' = full depth profile, 'surface' = surface only, 'bottom' = bottom only
        save_plot : bool
            Whether to save the plot (default: True)
        make_video : bool
            Whether to create video from multiple time steps (default: True)
        fps : int
            Frames per second for video (default: 4)
        quiver_n : int
            Subsample every n-th node for quivers (default: 3)
        quiver_m : int
            Subsample every m-th depth level for quivers (default: 10)
        """
        # Set output folder
        if output_folder is None:
            output_folder = extracted_folder
        
        plots_folder = os.path.join(output_folder, f'{river_branch}_plots', 'cross_section')
        os.makedirs(plots_folder, exist_ok=True)
        
        # Load nodes file if provided
        node_depths = None
        if nodes_file and os.path.exists(nodes_file):
            node_df = pd.read_csv(nodes_file)
            if 'depth' in node_df.columns:
                node_depths = node_df['depth'].values
                print(f"Loaded node depths from: {nodes_file}")
        
        # Determine which data to use
        has_hydro = self.config.plot_hydro and self.hydro_data is not None
        has_ts = self.config.plot_ts and self.ts_data is not None
        
        if has_ts and has_hydro:
            data = self.ts_data
            hydro_data = self.hydro_data
            var_list = [('salinity', self.config.salinity_levels, cmo.haline, '[g/kg]'),
                       ('temperature', self.config.temperature_levels, cmo.thermal, '°C')]
        elif has_ts and not has_hydro:
            data = self.ts_data
            hydro_data = None
            var_list = [('salinity', self.config.salinity_levels, cmo.haline, '[g/kg]'),
                       ('temperature', self.config.temperature_levels, cmo.thermal, '°C')]
        elif has_hydro and not has_ts:
            data = self.hydro_data
            hydro_data = None
            var_list = [('u_velocity', 20, 'RdBu_r', 'm/s'),
                       ('v_velocity', 20, 'RdBu_r', 'm/s')]
        else:
            print("No data loaded. Please load TS or hydro data first.")
            return
        
        # Parse time_idx
        if isinstance(time_idx, str) and time_idx.lower() == 'all':
            time_indices = range(len(data['time']))
        elif isinstance(time_idx, list) and len(time_idx) == 2:
            time_indices = range(time_idx[0], time_idx[1] + 1)
        elif isinstance(time_idx, int):
            time_indices = [time_idx]
        else:
            print(f"Invalid time_idx: {time_idx}")
            return
        
        # Set limits - use data limits if not provided
        if y_lims is None:
            y_lims = [data['Y'].min(), data['Y'].max()]
        if x_lims is None:
            x_lims = [data['distance'].min(), data['distance'].max()]
        
        # Store image paths for video
        image_paths = []
        
        # Plot each time step
        for t_idx in time_indices:
            if t_idx >= len(data['time']):
                print(f"Time index {t_idx} out of range (max: {len(data['time'])-1})")
                continue
            
            fig, axes = plt.subplots(len(var_list), 1, figsize=(14, 5 * len(var_list)))
            if len(var_list) == 1:
                axes = [axes]
            
            for ax, (var_name, levels, cmap, units) in zip(axes, var_list):
                # Get data for this variable
                var_data = data[var_name][t_idx, :, :].copy()
                
                # Get X and Y for plotting
                X = data['X']
                Y = data['Y'].copy()  # Copy to modify with water level
                
                # --- UPDATE Y WITH WATER LEVEL (if hydro data is available) ---
                if has_hydro and has_ts:
                    wl = hydro_data['water_level'][t_idx, :]
                    Y_interp = Y.copy()
                    for col in range(Y.shape[1]):
                        Y_interp[0, col] = wl[col]
                        Y_interp[1:, col] = Y[1:, col] + (wl[col] - Y[0, col])
                    Y = Y_interp
                    Y_modified = Y
                else:
                    Y_modified = Y
                
                # Determine layer selection
                if layer == 'surface':
                    layer_idx = -1
                    var_plot = var_data[:, layer_idx]
                    ax.plot(data['distance'], var_plot, 'b-', linewidth=2)
                    ax.set_ylabel(units)
                    
                elif layer == 'bottom':
                    layer_idx = 0
                    var_plot = var_data[:, layer_idx]
                    ax.plot(data['distance'], var_plot, 'b-', linewidth=2)
                    ax.set_ylabel(units)
                    
                else:  # 'all' - full depth profile
                    var_plot = var_data.T
                    var_plot[var_plot == 0] = np.nan
                    mask = np.ones_like(var_plot)
                    mask[np.isnan(var_plot)] = 0
                    
                    # --- UPDATE MASK WITH NODE DEPTHS (bottom boundary) ---
                    if node_depths is not None:
                        for col_idx in range(var_plot.shape[1]):
                            col_data = var_plot[:, col_idx]
                            valid_idx = np.where(~np.isnan(col_data))[0]
                            if len(valid_idx) > 0:
                                last_valid = valid_idx[-1]
                                node_depth = -node_depths[col_idx]
                                Y_modified[last_valid, col_idx] = node_depth
                    
                    # Plot contours
                    im = ax.contourf(X, Y_modified, var_plot, cmap=cmap, levels=levels, zorder=2)
                    plt.colorbar(im, ax=ax, label=units)
                    
                    # Add gray mask for NaN regions
                    ax.contourf(X, Y_modified, mask, colors='gray', 
                               levels=[-0.5, 0.5, 1.5], alpha=1.0, zorder=1)
                    
                    # Add bathymetry line
                    if node_depths is not None:
                        ax.plot(data['distance'], -node_depths, 'k-', linewidth=2, 
                               label='Bathymetry')
                    else:
                        ax.plot(data['distance'], data['total_depth'], 'k-', linewidth=2,
                               label='Bathymetry')
                    
                    # --- ADD WATER LEVEL LINE (if hydro data is available) ---
                    if has_hydro and has_ts:
                        ax.plot(data['distance'], wl, 'k--', linewidth=2, 
                               label='Water Level')
                    
                    # --- ADD QUIVERS (if hydro data is available) ---
                    if has_hydro and has_ts:
                        uv = hydro_data['u_velocity'][t_idx, :, :].T
                        vv = hydro_data['v_velocity'][t_idx, :, :].T
                        self._add_quivers_cross_section(ax, uv, vv, X, Y_modified, 
                                                        n=quiver_n, m=quiver_m)
                    
                    ax.invert_yaxis()
                    ax.legend(loc='lower right')
                
                # Set limits
                ax.set_xlim(x_lims)
                ax.set_ylim(y_lims)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Distance along transect (m)')
                ax.set_title(f'{var_name} at time {data["time"][t_idx]}')
            
            plt.tight_layout()
            
            # Save or show
            if save_plot:
                layer_suffix = f'_{layer}' if layer != 'all' else ''
                filename = f'cross_section_t{t_idx:04d}{layer_suffix}.png'
                filepath = save_figure(fig, plots_folder, filename, dpi=self.config.dpi)
                print(f"Cross-section plot saved: {filepath}")
                image_paths.append(filepath)
            else:
                plt.show()
            
            plt.close(fig)
        
        # Create video if multiple time steps and make_video is True
        if make_video and save_plot and len(image_paths) > 1:
            layer_suffix = f'_{layer}' if layer != 'all' else ''
            output_video = os.path.join(plots_folder, f'{river_branch}_cross_section{layer_suffix}_video.mp4')
            create_video(image_paths, output_video, fps=fps)
            print(f"Cross-section video saved at {output_video}")

# ============================================================================
# PLOT TYPE 2: 2D-V HOVMOLLER
# ============================================================================

    def plot_hovmoller(self, river_branch: str, output_folder: str,
                       variable: str = 'salinity',
                       layer: str = 'bottom',
                       threshold: Optional[float] = None,
                       x_tick_space: int = 15,
                       y_lims: Optional[List[float]] = None,
                       time_range: Optional[List[int]] = None,
                       save_plot: bool = True):
        """
        Plot Hovmoller diagram (time vs distance along transect).
        x-axis = time, y-axis = distance along transect.
        
        Parameters
        ----------
        river_branch : str
            Name of river branch
        output_folder : str
            Output folder for plots
        variable : str
            Variable to plot ('salinity' or 'temperature')
        layer : str
            'surface' or 'bottom' (bottom = last valid data point, not fixed index)
        threshold : float, optional
            Threshold value to contour (e.g., 35 for open sea, 2 for river)
        x_tick_space : int, optional
            Number of time steps between x-ticks (default: 15)
        y_lims : list, optional
            Y-axis limits [min, max] (distance along transect)
        time_range : list, optional
            Time range [start_idx, end_idx] to plot
        save_plot : bool
            Whether to save the plot (default: True)
        """
        plots_folder = os.path.join(output_folder, f'{river_branch}_plots', 'hovmoller')
        os.makedirs(plots_folder, exist_ok=True)
        
        if not self.config.plot_ts or self.ts_data is None:
            print("No tracer data available for Hovmoller plot")
            return
        
        data = self.ts_data
        time_vals = data['time']
        
        # Get layer data using the specialized functions
        if layer == 'surface':
            var_2d = self.get_surface_salt_branch(time_vals, data)
            layer_label = 'surface'
        else:  # 'bottom'
            var_2d = self.get_bottom_salt_branch(time_vals, data)
            layer_label = 'bottom'
        
        # Transpose to have time as rows, distance as columns
        var_2d = var_2d.T  # Now shape: (time, node)
        
        # Select time range
        if time_range is not None and len(time_range) == 2:
            t_start, t_end = time_range
            var_2d = var_2d[t_start:t_end, :]
            time_vals = time_vals[t_start:t_end]
        
        # Set y_lims (distance along transect)
        if y_lims is None:
            y_lims = [0, data['distance'].max()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Convert time to numeric indices for plotting
        time_numeric = np.arange(len(time_vals))
        
        # Plot Hovmoller
        im = ax.pcolormesh(time_numeric, data['distance'], var_2d.T,
                           cmap=cmo.haline if variable == 'salinity' else cmo.thermal,
                           shading='auto')
        
        plt.colorbar(im, ax=ax, label=f'{variable} ({layer_label})')
        
        # Set x-ticks with user-defined spacing
        if x_tick_space > 0:
            tick_indices = np.arange(0, len(time_vals), x_tick_space)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([t.strftime('%Y-%m-%d') for t in time_vals[tick_indices]], 
                              rotation=45, ha='right')
        
        # Add threshold contour if provided
        if threshold is not None:
            threshold_distances = []
            threshold_times = []
            threshold_time_indices = []
            
            for t_idx in range(var_2d.shape[0]):
                col_data = var_2d[t_idx, :]
                valid_idx = np.where(~np.isnan(col_data))[0]
                if len(valid_idx) > 0:
                    # Find where threshold is crossed
                    above_threshold = col_data > threshold
                    if np.any(above_threshold):
                        # Find first index where threshold is crossed
                        cross_idx = np.argmax(above_threshold)
                        if cross_idx < len(data['distance']):
                            threshold_distances.append(data['distance'][cross_idx])
                            threshold_times.append(time_vals[t_idx])
                            threshold_time_indices.append(t_idx)
            
            if len(threshold_distances) > 0:
                # Convert time indices for plotting
                threshold_time_numeric = np.array(threshold_time_indices)
                ax.plot(threshold_time_numeric, threshold_distances, 'r-', linewidth=2,
                       label=f'Threshold {threshold} {variable}')
                ax.plot(threshold_time_numeric, threshold_distances, 'ro', markersize=4)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance along transect (m)')
        ax.set_ylim(y_lims)
        ax.set_title(f'{variable} - {layer_label} layer (Hovmoller)')
        ax.grid(True, alpha=0.3)
        
        if threshold is not None:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_plot:
            threshold_suffix = f'_thr{threshold}' if threshold is not None else ''
            filename = f'hovmoller_{variable}_{layer}{threshold_suffix}.png'
            filepath = save_figure(fig, plots_folder, filename, dpi=self.config.dpi)
            print(f"Hovmoller plot saved: {filepath}")
        else:
            plt.show()
        
        plt.close(fig)


# ============================================================================
# PLOT TYPE 3: 2D-H MAP
# ============================================================================

    def plot_map(self, river_branch: str, grd_file: str,
             ts_file: Optional[str] = None,
             hydro_file: Optional[str] = None,
             output_folder: Optional[str] = None,
             time_idx: Union[int, List[int], str] = 0,
             x_lims: Optional[List[float]] = None,
             y_lims: Optional[List[float]] = None,
             layer: Union[str, int] = 'surface',
             plot_ts: bool = True,
             plot_hydro: bool = True,
             plot_quivers: bool = True,
             quiver_grid_resolution: int = 5,
             shapefiles: Optional[List[Dict[str, str]]] = None,
             dat_files: Optional[List[Union[str, Dict]]] = None,  # NEW
             save_plot: bool = True,
             make_video: bool = True,
             fps: int = 4,
             sal_limits: Optional[List[float]] = None,
             temp_limits: Optional[List[float]] = None,
             wl_limits: Optional[List[float]] = None,
             vel_limits: Optional[List[float]] = None):
        """
        Plot 2D horizontal map (birds-eye view) with optional overlays.
        
        Parameters
        ----------
        river_branch : str
            Name of river branch (used for output folder naming)
        grd_file : str
            Path to GRD file for triangulation
        ts_file : str, optional
            Path to TS NetCDF file (salinity, temperature)
        hydro_file : str, optional
            Path to Hydro NetCDF file (water level, velocities)
        output_folder : str, optional
            Output folder for plots
        time_idx : int, list, or str
            Time index(es) to plot
        x_lims : list, optional
            X-axis limits [min, max] (longitude)
        y_lims : list, optional
            Y-axis limits [min, max] (latitude)
        layer : str or int
            'surface', 'bottom', or layer index (0, 1, 2, ...)
        plot_ts : bool
            Plot salinity and temperature (default: True)
        plot_hydro : bool
            Plot water level and velocity magnitude (default: True)
        plot_quivers : bool
            Plot quivers on all subplots (default: True)
        shapefiles : list of dict, optional
            List of shapefiles to overlay
        save_plot : bool
            Whether to save the plot (default: True)
        make_video : bool
            Whether to create video (default: True)
        fps : int
            Frames per second for video (default: 4)
        sal_limits : list, optional
            Color limits for salinity [min, max]
        temp_limits : list, optional
            Color limits for temperature [min, max]
        wl_limits : list, optional
            Color limits for water level [min, max]
        vel_limits : list, optional
            Color limits for velocity magnitude [min, max]
        """
        # Set output folder
        if output_folder is None:
            if ts_file is not None:
                output_folder = os.path.dirname(ts_file)
            elif hydro_file is not None:
                output_folder = os.path.dirname(hydro_file)
            else:
                output_folder = os.getcwd()
        
        # Create output folders
        ts_folder = os.path.join(output_folder, f'{river_branch}_plots', 'maps_ts')
        hydro_folder = os.path.join(output_folder, f'{river_branch}_plots', 'maps_hydro')
        os.makedirs(ts_folder, exist_ok=True)
        os.makedirs(hydro_folder, exist_ok=True)
        
        # Load GRD for triangulation
        points, elements = self.read_grd(grd_file)
        
        if points and elements:
            point_ids = list(points.keys())
            id_to_index = {pid: idx for idx, pid in enumerate(point_ids)}
            grd_nodes = np.array([[points[pid]['x'], points[pid]['y']] for pid in point_ids])
            grd_elements = np.array([
                [id_to_index[nid] for nid in elem['node_ids']]
                for elem in elements
            ])
            triang = tri.Triangulation(grd_nodes[:, 0], grd_nodes[:, 1], grd_elements)
            print(f"Created triangulation with {len(grd_nodes)} nodes and {len(grd_elements)} elements")
        else:
            print("Failed to load GRD file")
            return
        
        # Set limits
        if x_lims is None:
            x_lims = [grd_nodes[:, 0].min(), grd_nodes[:, 0].max()]
        if y_lims is None:
            y_lims = [grd_nodes[:, 1].min(), grd_nodes[:, 1].max()]
        
        # Load TS data if requested and file provided
        has_ts = False
        ts_data = None
        if plot_ts and ts_file is not None and os.path.exists(ts_file):
            ts_data = self._load_ts_netcdf_full(ts_file)
            has_ts = True
            print(f"Loaded TS data from: {ts_file}")
        elif plot_ts:
            print("Warning: plot_ts=True but no TS file provided or file not found")
        
        # Load Hydro data if requested and file provided
        has_hydro = False
        hydro_data = None
        if plot_hydro and hydro_file is not None and os.path.exists(hydro_file):
            hydro_data = self._load_hydro_netcdf_full(hydro_file)
            has_hydro = True
            print(f"Loaded Hydro data from: {hydro_file}")
        elif plot_hydro:
            print("Warning: plot_hydro=True but no Hydro file provided or file not found")
        
        if not has_ts and not has_hydro:
            print("No data loaded. Please provide valid TS or Hydro files.")
            return
        
        # Get time values
        if has_ts:
            time_vals = ts_data['time']
            nc_nodes = len(ts_data['lon'])
            n_layers = ts_data['salinity'].shape[2]
        else:
            time_vals = hydro_data['time']
            nc_nodes = len(hydro_data['lon'])
            n_layers = hydro_data['u_velocity'].shape[2]
        
        # Check node count
        grd_nodes_count = len(grd_nodes)
        if nc_nodes != grd_nodes_count:
            print(f"WARNING: Node count mismatch! GRD has {grd_nodes_count} nodes, NetCDF has {nc_nodes} nodes.")
        
        # Determine layer index
        if isinstance(layer, str):
            if layer.lower() == 'surface':
                layer_idx = 0
                layer_name = 'surface'
            elif layer.lower() == 'bottom':
                layer_idx = -1  # Will be handled separately
                layer_name = 'bottom'
            else:
                layer_idx = 0
                layer_name = 'surface'
        else:
            layer_idx = int(layer)
            layer_name = f'layer{layer_idx:04d}'
        
        # Parse time_idx
        if isinstance(time_idx, str) and time_idx.lower() == 'all':
            time_indices = range(len(time_vals))
        elif isinstance(time_idx, list) and len(time_idx) == 2:
            time_indices = range(time_idx[0], time_idx[1] + 1)
        elif isinstance(time_idx, int):
            time_indices = [time_idx]
        else:
            print(f"Invalid time_idx: {time_idx}")
            return
        
        # Store image paths for videos
        ts_image_paths = []
        hydro_image_paths = []
        
        # Plot each time step
        for t_idx in time_indices:
            if t_idx >= len(time_vals):
                print(f"Time index {t_idx} out of range (max: {len(time_vals)-1})")
                continue
            
            # Initialize variables
            sal_data = None
            temp_data = None
            wl_data = None
            uv_data = None
            vv_data = None
            vel_mag = None
            
            # Prepare data for this time step
            if has_ts:
                if layer_name == 'bottom':
                    # Get bottom layer (last valid value for each node)
                    sal_full = ts_data['salinity'][t_idx, :, :]
                    temp_full = ts_data['temperature'][t_idx, :, :]
                    
                    # Find last valid index for each node
                    sal_data = np.full(sal_full.shape[0], np.nan)
                    temp_data = np.full(temp_full.shape[0], np.nan)
                    
                    for node_idx in range(sal_full.shape[0]):
                        # Find valid indices (non-zero and not NaN)
                        valid_idx = np.where((sal_full[node_idx, :] != 0) & 
                                            (~np.isnan(sal_full[node_idx, :])))[0]
                        if len(valid_idx) > 0:
                            sal_data[node_idx] = sal_full[node_idx, valid_idx[-1]]
                        
                        valid_idx = np.where((temp_full[node_idx, :] != 0) & 
                                            (~np.isnan(temp_full[node_idx, :])))[0]
                        if len(valid_idx) > 0:
                            temp_data[node_idx] = temp_full[node_idx, valid_idx[-1]]
                else:
                    # Surface or specific layer
                    sal_data = ts_data['salinity'][t_idx, :, layer_idx]
                    temp_data = ts_data['temperature'][t_idx, :, layer_idx]
            
            if has_hydro:
                if layer_name == 'bottom':
                    # Get bottom layer for hydro
                    uv_full = hydro_data['u_velocity'][t_idx, :, :]
                    vv_full = hydro_data['v_velocity'][t_idx, :, :]
                    
                    uv_data = np.full(uv_full.shape[0], np.nan)
                    vv_data = np.full(vv_full.shape[0], np.nan)
                    
                    for node_idx in range(uv_full.shape[0]):
                        valid_idx = np.where((uv_full[node_idx, :] != 0) & 
                                            (~np.isnan(uv_full[node_idx, :])))[0]
                        if len(valid_idx) > 0:
                            uv_data[node_idx] = uv_full[node_idx, valid_idx[-1]]
                        
                        valid_idx = np.where((vv_full[node_idx, :] != 0) & 
                                            (~np.isnan(vv_full[node_idx, :])))[0]
                        if len(valid_idx) > 0:
                            vv_data[node_idx] = vv_full[node_idx, valid_idx[-1]]
                    
                    wl_data = hydro_data['water_level'][t_idx, :]
                else:
                    wl_data = hydro_data['water_level'][t_idx, :]
                    uv_data = hydro_data['u_velocity'][t_idx, :, layer_idx]
                    vv_data = hydro_data['v_velocity'][t_idx, :, layer_idx]
                
                vel_mag = np.sqrt(uv_data**2 + vv_data**2)
            
            # Get time string for filename
            if hasattr(time_vals[t_idx], 'strftime'):
                time_str = time_vals[t_idx].strftime('%Y%m%d_%H%M')
            else:
                time_str = pd.Timestamp(time_vals[t_idx]).strftime('%Y%m%d_%H%M')
            
            # ============================================================
            # FIGURE 1: TS (Salinity + Temperature)
            # ============================================================
            if has_ts and plot_ts:
                fig_ts, axes_ts = plt.subplots(1, 2, figsize=(16, 7))
                
                # Salinity
                self._plot_map_panel(axes_ts[0], triang, sal_data, 
                                    'Salinity', 'g/kg',
                                    x_lims, y_lims,
                                    self.config.salinity_levels,
                                    cmo.haline,
                                    time_str, layer_name,
                                    sal_limits=sal_limits,
                                    has_hydro=has_hydro, 
                                    uv_data=uv_data, 
                                    vv_data=vv_data, 
                                    plot_quivers=plot_quivers,
                                    quiver_grid_resolution=quiver_grid_resolution,
                                    shapefiles=shapefiles,
                                    dat_files=dat_files)  
                
                # Temperature
                self._plot_map_panel(axes_ts[1], triang, temp_data,
                                    'Temperature', '°C',
                                    x_lims, y_lims,
                                    self.config.temperature_levels,
                                    cmo.thermal,
                                    time_str, layer_name,
                                    temp_limits=temp_limits,
                                    has_hydro=has_hydro, 
                                    uv_data=uv_data, 
                                    vv_data=vv_data, 
                                    plot_quivers=plot_quivers,
                                    quiver_grid_resolution=quiver_grid_resolution,
                                    shapefiles=shapefiles,
                                    dat_files=dat_files)  
                
                plt.tight_layout()
                
                if save_plot:
                    filename = f'map_ts_{time_str}_{layer_name}.png'
                    filepath = save_figure(fig_ts, ts_folder, filename, dpi=self.config.dpi)
                    ts_image_paths.append(filepath)
                else:
                    plt.show()
                plt.close(fig_ts)
            
            # ============================================================
            # FIGURE 2: Hydro (Water Level + Velocity Magnitude)
            # ============================================================
            if has_hydro and plot_hydro:
                fig_hydro, axes_hydro = plt.subplots(1, 2, figsize=(16, 7))
                
                # Water Level
                wl_levels = np.arange(-1.0, 1.1, 0.1)
                self._plot_map_panel(axes_hydro[0], triang, wl_data,
                                    'Water Level', 'm',
                                    x_lims, y_lims,
                                    wl_levels,
                                    cmo.balance,
                                    time_str, layer_name,
                                    wl_limits=wl_limits,
                                    has_hydro=has_hydro, 
                                    uv_data=uv_data, 
                                    vv_data=vv_data, 
                                    plot_quivers=plot_quivers,
                                    quiver_grid_resolution=quiver_grid_resolution,
                                    shapefiles=shapefiles,
                                    dat_files=dat_files)  
                
                # Velocity Magnitude
                vel_levels = np.arange(0, 1.0, 0.1)
                self._plot_map_panel(axes_hydro[1], triang, vel_mag,
                                    'Velocity Magnitude', 'm/s',
                                    x_lims, y_lims,
                                    vel_levels,
                                    cmo.speed,
                                    time_str, layer_name,
                                    vel_limits=vel_limits,
                                    has_hydro=has_hydro,
                                    uv_data=uv_data,
                                    vv_data=vv_data,
                                    plot_quivers=plot_quivers,
                                    quiver_grid_resolution=quiver_grid_resolution,
                                    shapefiles=shapefiles,
                                    dat_files=dat_files)  
                
                plt.tight_layout()
                
                if save_plot:
                    filename = f'map_hydro_{time_str}_{layer_name}.png'
                    filepath = save_figure(fig_hydro, hydro_folder, filename, dpi=self.config.dpi)
                    hydro_image_paths.append(filepath)
                else:
                    plt.show()
                plt.close(fig_hydro)
        
        # Create videos
        if make_video and save_plot:
            if ts_image_paths and plot_ts:
                output_video = os.path.join(ts_folder, f'{river_branch}_ts_{layer_name}_video.mp4')
                create_video(ts_image_paths, output_video, fps=fps)
                print(f"TS video saved at {output_video}")
            
            if hydro_image_paths and plot_hydro:
                output_video = os.path.join(hydro_folder, f'{river_branch}_hydro_{layer_name}_video.mp4')
                create_video(hydro_image_paths, output_video, fps=fps)
                print(f"Hydro video saved at {output_video}")
        
        print("Map plots complete!")