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
from dataclasses import dataclass
import cmocean.cm as cmo
import os
import numpy as np
from dataclasses import dataclass, field

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
    time_units: str = 'hours since 2022-07-15 00:00:00'  # Can be changed per file
    time_calendar: str = 'standard'
    date_range: Optional[Tuple[str, str]] = None  # e.g., ("2022-07-01", "2022-09-01")
    
    # Contour levels - FIXED with default_factory
    salinity_levels: np.ndarray = field(
        default_factory=lambda: np.array([0, 2, 5, 10, 15, 20, 25, 30, 35, 40])
    )
    temperature_levels: np.ndarray = field(
        default_factory=lambda: np.array([16, 18, 20, 22, 24, 26, 28, 30, 32])
    )
    
    # Output settings
    output_folder: Optional[str] = None
    fps: int = 4
    dpi: int = 100
    
    def __post_init__(self):
        """Validate settings after initialization."""
        # You can add validation here if needed
        pass
    
class RiverTransectPlotter:
    """Plotter for river transect data."""
    
    def __init__(self, config: RiverPlotConfig):
        """
        Initialize plotter with configuration.
        
        Parameters
        ----------
        config : RiverPlotConfig
            Plot configuration
        """
        self.config = config
        self.hydro_data = None
        self.ts_data = None
        self.discharge_dates = None
        self.discharge_values = None
        self.measurement_data = None
        self.store_swi = []
        self.image_paths = []
        
    def load_discharge_data(self, discharge_file: str):
        """Load discharge data from .dat file."""
        dates, values = self._read_dat_file(discharge_file)
        self.discharge_dates = dates
        self.discharge_values = values
    
    def load_measurement_data(self, measurement_files: dict, dist_points: list):
        """Load measurement data for multiple points."""
        self.measurement_data = {}
        for file_id, dist_point in zip(measurement_files.keys(), dist_points):
            self.measurement_data[file_id] = self._read_measurement_file(measurement_files[file_id])
    
    def load_model_data(self, models_folder: str, river_branch: str):
        """
        Load model data from extracted NetCDF files.
        
        Parameters
        ----------
        models_folder : str
            Path to folder containing extracted NetCDF files
        river_branch : str
            Name of river branch
        """
        if self.config.plot_hydro:
            hydro_file = f"{models_folder}/{river_branch}_hydro_extracted.nc"
            self.hydro_data = self._load_hydro_netcdf(hydro_file)
        
        if self.config.plot_ts:
            ts_file = f"{models_folder}/{river_branch}_ts_extracted.nc"
            self.ts_data = self._load_ts_netcdf(ts_file)
    
    def _read_dat_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read .dat file with discharge data."""
        dates = []
        values = []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                date_str = parts[0]
                value = float(parts[1])
                date = datetime.strptime(date_str, '%Y-%m-%d::%H:%M:%S')
                dates.append(date)
                values.append(value)
        return np.array(dates), np.array(values)
    
    def _read_measurement_file(self, filename: str) -> pd.DataFrame:
        """Read measurement file."""
        return pd.read_csv(
            filename,
            sep='\t',
            parse_dates=['Date'],
            date_format='%Y-%m-%d::%H:%M:%S'
        )
    
    def _load_hydro_netcdf(self, filename: str):
        """Load hydro NetCDF data."""
        ds = Dataset(filename)
        
        # Convert time
        t = num2date(ds.variables['time'][:], 
                     units=self.config.time_units, 
                     calendar=self.config.time_calendar)
        
        level = ds.variables['depth'][:]
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        td = ds.variables['total_depth'][:] * -1
        wl = ds.variables['water_level'][:]
        uv = ds.variables['u_velocity'][:]
        vv = ds.variables['v_velocity'][:]
        
        # Calculate distances
        dist = self._calculate_distances(lat, lon)
        
        # Create meshgrid
        X, Y = np.meshgrid(dist, level * -1)
        
        # Water level dates
        wl_dates = np.array([datetime(t_i.year, t_i.month, t_i.day, 
                                       t_i.hour, t_i.minute, t_i.second) 
                            for t_i in t])
        wl_values = wl[:, -1]
        
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
            'wl_dates': wl_dates,
            'wl_values': wl_values,
            'X': X,
            'Y': Y,
            'distance': dist
        }
    
    def _load_ts_netcdf(self, filename: str):
        """Load tracer (salinity/temperature) NetCDF data."""
        ds = Dataset(filename)
        
        sal = ds.variables['salinity'][:]
        tem = ds.variables['temperature'][:]
        
        # Convert time
        t = num2date(ds.variables['time'][:], 
                     units=self.config.time_units, 
                     calendar=self.config.time_calendar)
        
        level = ds.variables['depth'][:]
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        td = ds.variables['total_depth'][:] * -1
        
        # Calculate distances
        dist = self._calculate_distances(lat, lon)
        dist = np.flipud(dist)
        
        # Create meshgrid
        X, Y = np.meshgrid(dist, level * -1)
        
        # Time array
        ts_dates = np.array([datetime(t_i.year, t_i.month, t_i.day,
                                       t_i.hour, t_i.minute, t_i.second)
                            for t_i in t])
        
        ds.close()
        
        return {
            'time': t,
            'level': level,
            'lon': lon,
            'lat': lat,
            'total_depth': td,
            'salinity': sal,
            'temperature': tem,
            'ts_dates': ts_dates,
            'X': X,
            'Y': Y,
            'distance': dist
        }
    
    def _calculate_distances(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Calculate cumulative distances along river nodes."""
        c_d = [haversine(lat[i], lon[i], lat[i + 1], lon[i + 1]) 
               for i in range(len(lat) - 1)]
        c_dd = [i * 1000 for i in c_d]
        c_d = [0] + c_dd
        return np.cumsum(c_d)
    
    def _detect_threshold_exceedance(self, c_sal, X, threshold=2):
        """Find first column where salinity exceeds threshold."""
        for col in range(c_sal.shape[1]):
            if np.any(c_sal[:, col] > threshold):
                return X[:, col], col
        return None, None
    
    def plot(self, river_branch: str, sims_name: str, output_folder: str):
        """
        Generate all plots and video.
        
        Parameters
        ----------
        river_branch : str
            Name of river branch
        sims_name : str
            Simulation name
        output_folder : str
            Output folder for plots and video
        """
        # Create output folders
        plots_folder = os.path.join(output_folder, f'{river_branch}_plots', 'vertical_plots')
        os.makedirs(plots_folder, exist_ok=True)
        
        # Determine time array
        if self.config.plot_ts:
            t = self.ts_data['time']
        else:
            t = self.hydro_data['time']
        
        # Date range for x-axis
        if self.config.date_range:
            start_date = pd.to_datetime(self.config.date_range[0])
            end_date = pd.to_datetime(self.config.date_range[1])
        else:
            start_date = t[0]
            end_date = t[-1]
        
        # Generate plots for each time step
        for ii in range(1, len(t), 1):
            c_t = t[ii]
            discharge_index = np.searchsorted(self.discharge_dates, c_t)
            
            # Extract current data
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
            
            # Calculate Y1 (depth profile)
            if self.config.plot_hydro:
                td = self.hydro_data['total_depth'].reshape(1, -1)
                Y1 = np.vstack((self.hydro_data['water_level'][ii, :], 
                                self.hydro_data['Y'][:, :]))
                middle_layers = (Y1[:-1, :] + Y1[1:, :]) / 2
                Y1 = np.vstack((self.hydro_data['water_level'][ii, :], middle_layers[:-1, :]))
            else:
                Y1 = self.ts_data['Y'][:, :] + 0.25
            
            # Create figure
            fig = self._create_figure(c_t, c_sal, c_tem, c_uv, c_vv, c_m, c_wl,
                                     Y1, X_length, col_X, discharge_index,
                                     start_date, end_date)
            
            # Save figure
            suffix = self._get_suffix()
            figure_name = f'plot_{ii:04d}{suffix}.png'
            filepath = save_figure(fig, plots_folder, figure_name, 
                                   dpi=self.config.dpi, close=True)
            self.image_paths.append(filepath)
        
        # Save SWI data if any
        if self.store_swi:
            self._save_swi_data(output_folder, river_branch)
        
        # Create video
        if self.config.make_video:
            self._create_video(output_folder, river_branch)
    
    def _create_figure(self, c_t, c_sal, c_tem, c_uv, c_vv, c_m, c_wl,
                       Y1, X_length, col_X, discharge_index,
                       start_date, end_date):
        """Create the figure with appropriate subplots based on configuration."""
        fig = plt.figure(figsize=(30, 15))
        gs = gridspec.GridSpec(9, 1)
        
        if self.config.plot_hydro and not self.config.plot_ts:
            # Case 1: hydro only
            ax1 = fig.add_subplot(gs[0:4, :])
            ax2 = fig.add_subplot(gs[4:8, :])
            ax3 = fig.add_subplot(gs[8:9, :])
            
            # Plot velocity magnitude
            surf_mag = ax1.contourf(self.hydro_data['X'], Y1, c_m, 
                                    cmap='viridis', levels=20, zorder=2)
            plt.colorbar(surf_mag, ax=ax1, label='Velocity Magnitude (m/s)')
            
            # Plot u_velocity
            surf_u = ax2.contourf(self.hydro_data['X'], Y1, c_uv,
                                  cmap='RdBu_r', levels=20, zorder=2)
            plt.colorbar(surf_u, ax=ax2, label='U Velocity (m/s)')
            
            # Plot water level
            ax1.plot(self.hydro_data['X'][0, :], self.hydro_data['water_level'][:, -1], 
                    'k', linewidth=4)
            ax2.plot(self.hydro_data['X'][0, :], self.hydro_data['water_level'][:, -1], 
                    'k', linewidth=4)
            
            # Plot bathymetry
            td = self.hydro_data['total_depth']
            ax1.fill_between(self.hydro_data['X'][0, :], -td, 
                            self.hydro_data['water_level'][:, -1], 
                            color='lightgray', alpha=0.5)
            ax2.fill_between(self.hydro_data['X'][0, :], -td,
                            self.hydro_data['water_level'][:, -1],
                            color='lightgray', alpha=0.5)
            
            # Add quivers
            self._add_quivers(ax1, c_uv, c_vv, self.hydro_data['X'], Y1)
            
        elif not self.config.plot_hydro and self.config.plot_ts:
            # Case 2: ts only
            ax1 = fig.add_subplot(gs[0:4, :])
            ax2 = fig.add_subplot(gs[4:8, :])
            ax3 = fig.add_subplot(gs[8:9, :])
            
            # Salinity plot
            surf_s = ax1.contourf(self.ts_data['X'], Y1, c_sal,
                                  cmap=cmo.haline, levels=self.config.salinity_levels,
                                  vmin=0, vmax=40, zorder=2)
            self._add_mask(ax1, c_sal, self.ts_data['X'], Y1)
            plt.colorbar(surf_s, ax=ax1, label='Salinity (PSU)')
            
            # Temperature plot
            surf_t = ax2.contourf(self.ts_data['X'], Y1, c_tem,
                                  cmap=cmo.thermal, levels=self.config.temperature_levels,
                                  vmin=16, vmax=32, zorder=2)
            self._add_mask(ax2, c_tem, self.ts_data['X'], Y1)
            plt.colorbar(surf_t, ax=ax2, label='Temperature (°C)')
            
            # Add contour lines
            contour_colors = ['w', 'w', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
            ax1.contour(self.ts_data['X'], Y1, c_sal, colors=contour_colors,
                       levels=self.config.salinity_levels, linewidths=2)
            ax2.contour(self.ts_data['X'], Y1, c_tem, colors=contour_colors,
                       levels=self.config.temperature_levels, linewidths=2)
            
            # SWI detection
            if X_length is not None and col_X is not None:
                self._add_swi_line(ax1, X_length, col_X, c_t, Y1)
            
        elif self.config.plot_hydro and self.config.plot_ts:
            # Case 3: both hydro and ts
            ax1 = fig.add_subplot(gs[0:4, :])
            ax2 = fig.add_subplot(gs[4:8, :])
            ax3 = fig.add_subplot(gs[8:9, :])
            
            # Salinity plot
            surf_s = ax1.contourf(self.ts_data['X'], Y1, c_sal,
                                  cmap=cmo.haline, levels=self.config.salinity_levels,
                                  vmin=0, vmax=40, zorder=2)
            self._add_mask(ax1, c_sal, self.ts_data['X'], Y1)
            plt.colorbar(surf_s, ax=ax1, label='Salinity (PSU)')
            
            # Temperature plot
            surf_t = ax2.contourf(self.ts_data['X'], Y1, c_tem,
                                  cmap=cmo.thermal, levels=self.config.temperature_levels,
                                  vmin=16, vmax=32, zorder=2)
            self._add_mask(ax2, c_tem, self.ts_data['X'], Y1)
            plt.colorbar(surf_t, ax=ax2, label='Temperature (°C)')
            
            # Add contour lines
            contour_colors = ['w', 'w', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
            ax1.contour(self.ts_data['X'], Y1, c_sal, colors=contour_colors,
                       levels=self.config.salinity_levels, linewidths=2)
            ax2.contour(self.ts_data['X'], Y1, c_tem, colors=contour_colors,
                       levels=self.config.temperature_levels, linewidths=2)
            
            # Plot water level
            ax1.plot(self.hydro_data['X'][0, :], c_wl,'k', linewidth=4)
            ax2.plot(self.hydro_data['X'][0, :], c_wl,'k', linewidth=4)
            
            # Plot bathymetry
            td = self.hydro_data['total_depth']
            # ax1.fill_between(self.hydro_data['X'][0, :], -td,
            #                 self.hydro_data['water_level'][:, -1],
            #                 color='lightgray', alpha=0.3)
            # ax2.fill_between(self.hydro_data['X'][0, :], -td,
            #                 self.hydro_data['water_level'][:, -1],
            #                 color='lightgray', alpha=0.3)
            
            # Add quivers
            self._add_quivers(ax1, c_uv, c_vv, self.hydro_data['X'], Y1)
            self._add_quivers(ax2, c_uv, c_vv, self.hydro_data['X'], Y1)
            
            # SWI detection
            if X_length is not None and col_X is not None:
                self._add_swi_line(ax1, X_length, col_X, c_t, Y1)
        
        # Add measurements if enabled
        if self.config.plot_measurements and self.config.plot_ts and self.measurement_data:
            self._add_measurements(ax1, c_t)
        
        # Add max SWI line if we have stored data
        if self.store_swi and self.config.plot_ts:
            self._add_max_swi_line(ax1, Y1)
        
        # Common axis formatting
        xticks = np.arange(0, 50001, 5000)
        for ax in [ax1, ax2]:
            ax.set_xlim(self.config.x_lim)
            ax.set_ylim(self.config.y_lim)
            ax.set_xticks(xticks)
            ax.set_ylabel('Depth (m)', fontsize=18)
            ax.tick_params(labelsize=20)
            ax.invert_xaxis()
        
        # Discharge plot
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
        """Add quiver arrows to axis."""
        magnitude = np.sqrt(c_uv**2 + c_vv**2)
        un = c_uv / magnitude
        vn = c_vv / magnitude
        un = np.nan_to_num(un)
        vn = np.nan_to_num(vn)
        
        u_direction = np.sign(c_uv[0::n, 0::m])
        v_direction = np.zeros_like(u_direction)
        
        mask_right = u_direction > 0
        mask_left = u_direction < 0
        X_flipped = np.fliplr(X)  # if X is 2D
        
        ax.quiver(X_flipped[0::n, 0::m][mask_right], Y1[0::n, 0::m][mask_right],
                  u_direction[mask_right], v_direction[mask_right],
                  scale=6, scale_units='inches', width=0.002, headwidth=4,
                  color='black', label='Out', zorder=3)
        ax.quiver(X_flipped[0::n, 0::m][mask_left], Y1[0::n, 0::m][mask_left],
                  u_direction[mask_left], v_direction[mask_left],
                  scale=6, scale_units='inches', width=0.002, headwidth=4,
                  color='blue', label='In', zorder=3)
    
    def _add_mask(self, ax, data, X, Y1):
        """Add gray mask for NaN regions."""
        mask = np.ones_like(data)
        mask[np.isnan(data)] = 0
        ax.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5],
                    alpha=1.0, zorder=1)
    
    def _add_swi_line(self, ax, X_length, col_X, c_t, Y1):
        """Add Salt Wedge Indicator line."""
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
        """Add line for maximum SWI length."""
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
        """Add measurement points to plot."""
        for file_id, dist_point in zip(self.measurement_data.keys(), [48591, 48499, 46915, 44004, 41644, 38871, 35719]):
            data = self.measurement_data[file_id]
            c_dist = np.mean(data['Distance'])
            
            if self.config.plot_hydro:
                closest_idx = np.abs(self.hydro_data['distance'] - c_dist).argmin()
                closest_wl = self.hydro_data['water_level'][:, closest_idx]
            else:
                closest_idx = np.abs(self.ts_data['distance'] - c_dist).argmin()
                closest_wl = 0
            
            # Insert new depth level at the water level
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
        """Get filename suffix based on plot configuration."""
        if self.config.plot_hydro and self.config.plot_ts:
            return '_hydro_ts'
        elif self.config.plot_hydro:
            return '_hydro'
        elif self.config.plot_ts:
            return '_ts'
        return ''
    
    def _save_swi_data(self, output_folder: str, river_branch: str):
        """Save SWI data to text file."""
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
        """Create video from saved images."""
        suffix = self._get_suffix()
        output_video = os.path.join(output_folder, f'{river_branch}_plots',
                                   f'{river_branch}{suffix}_video4.mp4')
        create_video(self.image_paths, output_video, fps=self.config.fps)
        print(f"Video saved at {output_video}")