# -*- coding: utf-8 -*-
"""
River transect plotting functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset, num2date
from datetime import datetime
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field
import cmocean.cm as cmo
import os

from ..io.nc_node_extractor import SHYFEMNodeExtractor
from .utils import haversine, save_figure, create_video


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
        default_factory=lambda: np.array([4,6,8,10,12,14,16, 18, 20, 22, 24, 26, 28, 30, 32,34,36,38,40])
    )
    
    # Output settings
    output_folder: Optional[str] = None
    fps: int = 4
    dpi: int = 100
    
    def __post_init__(self):
        pass


class RiverTransectPlotter:
    """Plotter for river transect data."""
    
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
    
    def _calculate_distances(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        c_d = [haversine(lat[i], lon[i], lat[i + 1], lon[i + 1]) 
               for i in range(len(lat) - 1)]
        c_dd = [i * 1000 for i in c_d]
        c_d = [0] + c_dd
        return np.cumsum(c_d)
    
    def _detect_threshold_exceedance(self, c_sal, X, threshold=2):
        for col in range(c_sal.shape[1]):
            if np.any(c_sal[:, col] > threshold):
                return X[:, col], col
        return None, None
    
    # ============================================================
# PLOT TYPE 1: 2D-V Cross-section (x-axis = distance along transect)
# ============================================================

    def plot_cross_section(self, river_branch: str, extracted_folder: str, 
                           output_folder: Optional[str] = None,
                           nodes_file: Optional[str] = None,
                           time_idx: Union[int, List[int], str] = 0,
                           y_lims: Optional[List[float]] = None,
                           x_lims: Optional[List[float]] = None,
                           layer: str = 'all', 
                           save_plot: bool = True,
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
            # Both TS and Hydro: use TS for contours, Hydro for WL and quivers
            data = self.ts_data
            hydro_data = self.hydro_data
            var_list = [('salinity', self.config.salinity_levels, cmo.haline, '[g/kg]'),
                       ('temperature', self.config.temperature_levels, cmo.thermal, '°C')]
        elif has_ts and not has_hydro:
            # TS only
            data = self.ts_data
            hydro_data = None
            var_list = [('salinity', self.config.salinity_levels, cmo.haline, '[g/kg]'),
                       ('temperature', self.config.temperature_levels, cmo.thermal, '°C')]
        elif has_hydro and not has_ts:
            # Hydro only
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
                    # Get water level for this time step
                    wl = hydro_data['water_level'][t_idx, :]
                    
                    # Update Y: replace first row (surface) with water level
                    # Y[0, :] = wl
                    # Add water level as new top row
                    # Y_new = np.vstack([wl, Y])
                    # Get mean of each row (if you want to average across nodes)
                    # row_means = np.mean(Y_new, axis=1)
                    # Option 3: Interpolate water level across all rows (same shape)
                    Y_interp = Y.copy()
                    for col in range(Y.shape[1]):
                        # Linear interpolation from surface (wl) to first depth level
                        Y_interp[0, col] = wl[col]
                        # Optionally interpolate the rest
                        Y_interp[1:, col] = Y[1:, col] + (wl[col] - Y[0, col])
                    Y=Y_interp
                    # Also update the original Y array in data for consistency
                    # (but we'll use the modified Y for plotting)
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
                    # Transpose data for contourf (X, Y)
                    var_plot = var_data.T  # Now shape: (depth, node)
                    
                    # Replace 0 values with NaN (outside domain)
                    var_plot[var_plot == 0] = np.nan
                    
                    # Create mask for NaN values
                    mask = np.ones_like(var_plot)
                    mask[np.isnan(var_plot)] = 0
                    
                    # --- UPDATE MASK WITH NODE DEPTHS (bottom boundary) ---
                    if node_depths is not None:
                        for col_idx in range(var_plot.shape[1]):
                            # Find the last valid index in this column (where data exists)
                            col_data = var_plot[:, col_idx]
                            valid_idx = np.where(~np.isnan(col_data))[0]
                            
                            if len(valid_idx) > 0:
                                last_valid = valid_idx[-1]  # Last index with data
                                
                                # Get the node depth for this column
                                node_depth = -node_depths[col_idx]
                                
                                # Update Y_modified at the last valid position with node_depth
                                Y_modified[last_valid, col_idx] = node_depth
                    
                    # Plot contours
                    im = ax.contourf(X, Y_modified, var_plot, cmap=cmap, levels=levels, zorder=2)
                    plt.colorbar(im, ax=ax, label=units)
                    
                    # Add gray mask for NaN regions
                    ax.contourf(X, Y_modified, mask, colors='gray', 
                               levels=[-0.5, 0.5, 1.5], alpha=1.0, zorder=1)
                    
                    # Add bathymetry line (from nodes file or total_depth)
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
                        # Get u/v velocities for this time step
                        uv = hydro_data['u_velocity'][t_idx, :, :].T  # Transpose
                        vv = hydro_data['v_velocity'][t_idx, :, :].T  # Transpose
                        
                        # Add quivers to the plot
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
            else:
                plt.show()
            
            plt.close(fig)
            
    # ============================================================
    # PLOT TYPE 2: 2D-V Time (Hovmoller, x-axis = time)
    # ============================================================
    
    def plot_hovmoller(self, river_branch: str, output_folder: str,
                       variable: str = 'salinity', layer: str = 'surface',
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
            'surface' or 'bottom'
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
        var_data = data[variable]
        
        # Select layer
        layer_idx = -1 if layer == 'surface' else 0
        var_2d = var_data[:, :, layer_idx]  # (time, node)
        
        # Select time range
        if time_range is not None and len(time_range) == 2:
            t_start, t_end = time_range
            var_2d = var_2d[t_start:t_end, :]
            time_vals = data['time'][t_start:t_end]
        else:
            time_vals = data['time']
        
        # Set y_lims
        if y_lims is None:
            y_lims = [0, data['distance'].max()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.pcolormesh(time_vals, data['distance'], var_2d.T,
                          cmap=cmo.haline if variable == 'salinity' else cmo.thermal,
                          shading='auto')
        
        plt.colorbar(im, ax=ax, label=f'{variable} ({layer})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance along transect (m)')
        ax.set_ylim(y_lims)
        ax.set_title(f'{variable} - {layer} layer (Hovmoller)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'hovmoller_{variable}_{layer}.png'
            filepath = save_figure(fig, plots_folder, filename, dpi=self.config.dpi)
            print(f"Hovmoller plot saved: {filepath}")
        else:
            plt.show()
        
        plt.close(fig)
    
    # ============================================================
    # PLOT TYPE 3: 2D-H Map (horizontal view)
    # ============================================================
    
    def plot_map(self, river_branch: str, output_folder: str,
                 variable: str = 'salinity', time_idx: int = 0, layer: str = 'surface'):
        """
        Plot 2D horizontal map (birds-eye view).
        
        Parameters
        ----------
        river_branch : str
            Name of river branch
        output_folder : str
            Output folder for plots
        variable : str
            Variable to plot ('salinity', 'temperature', 'water_level', 'u_velocity')
        time_idx : int
            Time index to plot
        layer : str
            'surface' or 'bottom'
        """
        plots_folder = os.path.join(output_folder, f'{river_branch}_plots', 'maps')
        os.makedirs(plots_folder, exist_ok=True)
        
        # Get data
        if variable in ['salinity', 'temperature']:
            data = self.ts_data
            var_data = data[variable][time_idx, :, :]
        else:
            data = self.hydro_data
            var_data = data[variable][time_idx, :, :]
        
        # Select layer
        if layer == 'surface':
            layer_idx = -1
        else:
            layer_idx = 0
        
        var_1d = var_data[:, layer_idx]
        
        # Get coordinates
        lon = data['lon']
        lat = data['lat']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot as scatter (for unstructured grid)
        sc = ax.scatter(lon, lat, c=var_1d, cmap=cmo.haline if 'sal' in variable else cmo.thermal,
                       s=30, edgecolors='k', linewidth=0.3)
        
        plt.colorbar(sc, ax=ax, label=f'{variable} ({layer})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{variable} - {layer} layer at time {data["time"][time_idx]}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        filename = f'map_{variable}_{layer}_t{time_idx:04d}.png'
        save_figure(fig, plots_folder, filename, dpi=self.config.dpi)
        print(f"Map plot saved: {filename}")
    
    # ============================================================
    # ORIGINAL PLOT FUNCTION (kept for backward compatibility)
    # ============================================================
    
    def plot(self, river_branch: str, sims_name: str, output_folder: str):
        """Original plot function - generates all time steps as before."""
        plots_folder = os.path.join(output_folder, f'{river_branch}_plots', 'vertical_plots')
        os.makedirs(plots_folder, exist_ok=True)
        
        if self.config.plot_ts:
            t = self.ts_data['time']
        else:
            t = self.hydro_data['time']
        
        if self.config.date_range:
            start_date = pd.to_datetime(self.config.date_range[0])
            end_date = pd.to_datetime(self.config.date_range[1])
        else:
            start_date = t[0]
            end_date = t[-1]
        
        for ii in range(1, len(t), 1):
            c_t = t[ii]
            discharge_index = np.searchsorted(self.discharge_dates, c_t) if self.discharge_dates is not None else 0
            
            if self.config.plot_hydro:
                c_uv = np.transpose(self.hydro_data['u_velocity'][ii, :, :])
                c_vv = np.transpose(self.hydro_data['v_velocity'][ii, :, :])
                c_uv[c_uv == 0] = np.nan
                c_vv[c_vv == 0] = np.nan
                c_m = np.sqrt(c_uv**2 + c_vv**2)
                c_wl = np.flipud(self.hydro_data['water_level'][ii,:])
            
            if self.config.plot_ts:
                c_sal = np.transpose(self.ts_data['salinity'][ii, :, :])
                c_tem = np.transpose(self.ts_data['temperature'][ii, :, :])
                c_sal[c_sal == 0] = np.nan
                c_tem[c_tem == 0] = np.nan
                X_length, col_X = self._detect_threshold_exceedance(c_sal, self.ts_data['X'])
            
            if self.config.plot_hydro:
                Y1 = np.vstack((self.hydro_data['water_level'][ii, :], 
                                self.hydro_data['Y'][:, :]))
                middle_layers = (Y1[:-1, :] + Y1[1:, :]) / 2
                Y1 = np.vstack((self.hydro_data['water_level'][ii, :], middle_layers[:-1, :]))
            else:
                Y1 = self.ts_data['Y'][:, :] + 0.25
            
            fig = self._create_figure(c_t, c_sal, c_tem, c_uv, c_vv, c_m, c_wl,
                                     Y1, X_length, col_X, discharge_index,
                                     start_date, end_date)
            
            suffix = self._get_suffix()
            figure_name = f'plot_{ii:04d}{suffix}.png'
            filepath = save_figure(fig, plots_folder, figure_name, 
                                   dpi=self.config.dpi, close=True)
            self.image_paths.append(filepath)
        
        if self.store_swi:
            self._save_swi_data(output_folder, river_branch)
        
        if self.config.make_video:
            self._create_video(output_folder, river_branch)
    
    def _create_figure(self, c_t, c_sal, c_tem, c_uv, c_vv, c_m, c_wl,
                       Y1, X_length, col_X, discharge_index,
                       start_date, end_date):
        """Original figure creation function (kept for backward compatibility)."""
        fig = plt.figure(figsize=(30, 15))
        gs = gridspec.GridSpec(9, 1)
        
        if self.config.plot_hydro and not self.config.plot_ts:
            ax1 = fig.add_subplot(gs[0:4, :])
            ax2 = fig.add_subplot(gs[4:8, :])
            ax3 = fig.add_subplot(gs[8:9, :])
            
            surf_mag = ax1.contourf(self.hydro_data['X'], Y1, c_m, 
                                    cmap='viridis', levels=20, zorder=2)
            plt.colorbar(surf_mag, ax=ax1, label='Velocity Magnitude (m/s)')
            surf_u = ax2.contourf(self.hydro_data['X'], Y1, c_uv,
                                  cmap='RdBu_r', levels=20, zorder=2)
            plt.colorbar(surf_u, ax=ax2, label='U Velocity (m/s)')
            
            ax1.plot(self.hydro_data['X'][0, :], self.hydro_data['water_level'][:, -1], 'k', linewidth=4)
            ax2.plot(self.hydro_data['X'][0, :], self.hydro_data['water_level'][:, -1], 'k', linewidth=4)
            
            td = self.hydro_data['total_depth']
            ax1.fill_between(self.hydro_data['X'][0, :], -td, 
                            self.hydro_data['water_level'][:, -1], color='lightgray', alpha=0.5)
            ax2.fill_between(self.hydro_data['X'][0, :], -td,
                            self.hydro_data['water_level'][:, -1], color='lightgray', alpha=0.5)
            self._add_quivers(ax1, c_uv, c_vv, self.hydro_data['X'], Y1)
            
        elif not self.config.plot_hydro and self.config.plot_ts:
            ax1 = fig.add_subplot(gs[0:4, :])
            ax2 = fig.add_subplot(gs[4:8, :])
            ax3 = fig.add_subplot(gs[8:9, :])
            
            surf_s = ax1.contourf(self.ts_data['X'], Y1, c_sal,
                                  cmap=cmo.haline, levels=self.config.salinity_levels,
                                  vmin=0, vmax=40, zorder=2)
            self._add_mask(ax1, c_sal, self.ts_data['X'], Y1)
            plt.colorbar(surf_s, ax=ax1, label='Salinity [g/kg]')
            
            surf_t = ax2.contourf(self.ts_data['X'], Y1, c_tem,
                                  cmap=cmo.thermal, levels=self.config.temperature_levels,
                                  vmin=16, vmax=32, zorder=2)
            self._add_mask(ax2, c_tem, self.ts_data['X'], Y1)
            plt.colorbar(surf_t, ax=ax2, label='Temperature (°C)')
            
            contour_colors = ['w', 'w', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
            ax1.contour(self.ts_data['X'], Y1, c_sal, colors=contour_colors,
                       levels=self.config.salinity_levels, linewidths=2)
            ax2.contour(self.ts_data['X'], Y1, c_tem, colors=contour_colors,
                       levels=self.config.temperature_levels, linewidths=2)
            
            if X_length is not None and col_X is not None:
                self._add_swi_line(ax1, X_length, col_X, c_t, Y1)
            
        elif self.config.plot_hydro and self.config.plot_ts:
            ax1 = fig.add_subplot(gs[0:4, :])
            ax2 = fig.add_subplot(gs[4:8, :])
            ax3 = fig.add_subplot(gs[8:9, :])
            
            surf_s = ax1.contourf(self.ts_data['X'], Y1, c_sal,
                                  cmap=cmo.haline, levels=self.config.salinity_levels,
                                  vmin=0, vmax=40, zorder=2)
            self._add_mask(ax1, c_sal, self.ts_data['X'], Y1)
            plt.colorbar(surf_s, ax=ax1, label='Salinity [g/kg]')
            
            surf_t = ax2.contourf(self.ts_data['X'], Y1, c_tem,
                                  cmap=cmo.thermal, levels=self.config.temperature_levels,
                                  vmin=16, vmax=32, zorder=2)
            self._add_mask(ax2, c_tem, self.ts_data['X'], Y1)
            plt.colorbar(surf_t, ax=ax2, label='Temperature (°C)')
            
            contour_colors = ['w', 'w', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
            ax1.contour(self.ts_data['X'], Y1, c_sal, colors=contour_colors,
                       levels=self.config.salinity_levels, linewidths=2)
            ax2.contour(self.ts_data['X'], Y1, c_tem, colors=contour_colors,
                       levels=self.config.temperature_levels, linewidths=2)
            
            ax1.plot(self.hydro_data['X'][0, :], c_wl, 'k', linewidth=4)
            ax2.plot(self.hydro_data['X'][0, :], c_wl, 'k', linewidth=4)
            
            self._add_quivers(ax1, c_uv, c_vv, self.hydro_data['X'], Y1)
            self._add_quivers(ax2, c_uv, c_vv, self.hydro_data['X'], Y1)
            
            if X_length is not None and col_X is not None:
                self._add_swi_line(ax1, X_length, col_X, c_t, Y1)
        
        if self.config.plot_measurements and self.config.plot_ts and self.measurement_data:
            self._add_measurements(ax1, c_t)
        
        if self.store_swi and self.config.plot_ts:
            self._add_max_swi_line(ax1, Y1)
        
        xticks = np.arange(0, 50001, 5000)
        for ax in [ax1, ax2]:
            ax.set_xlim(self.config.x_lim)
            ax.set_ylim(self.config.y_lim)
            ax.set_xticks(xticks)
            ax.set_ylabel('Depth (m)', fontsize=18)
            ax.tick_params(labelsize=20)
            ax.invert_xaxis()
        
        if self.discharge_dates is not None:
            ax3.plot(self.discharge_dates, self.discharge_values, 'k')
            if 0 <= discharge_index < len(self.discharge_dates):
                ax3.plot(self.discharge_dates[discharge_index], 
                        self.discharge_values[discharge_index], 'ro')
            ax3.set_xlim([start_date, end_date])
            ax3.set_ylim([0, self.config.max_discharge_lim])
            ax3.set_ylabel('Discharge (m³/s)')
            ax3.grid(True)
        
        fig.tight_layout()
        return fig
    
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
        """
        Add quiver arrows to cross-section plot.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to add quivers to
        c_uv : numpy.ndarray
            U velocity data (node, depth)
        c_vv : numpy.ndarray
            V velocity data (node, depth)
        X : numpy.ndarray
            X meshgrid (distance, depth)
        Y : numpy.ndarray
            Y meshgrid (distance, depth)
        n : int
            Subsample every n-th node
        m : int
            Subsample every m-th depth level
        """
        # Subsample the data
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
        mask_out = (uv_sub > 0) & valid  # Seaward (positive u)
        mask_in = (uv_sub < 0) & valid   # Landward (negative u)
        
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