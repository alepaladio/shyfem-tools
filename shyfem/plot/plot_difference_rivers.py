"-*- coding: utf-8 -*-"
"""
Created on Tue Jul  9 21:06:08 2024

@author: a_p_h
"""

import pandas as pd  # save as CSV
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import os
import imageio.v2 as imageio #import imageio
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from datetime import datetime
from typing import NamedTuple
import cmocean.cm as cmo
# %%
# Function to get the depth on each node based on the GRD file
def get_last_values(c_sal, Y):
    """
    Find the last valid index for each column in `c_sal` and retrieve the corresponding value from `Y`.
    Parameters:
    c_sal (numpy.ndarray): A 2D array (41, 422) containing values.
    Y (numpy.ndarray): A 2D array (41, 422) of the same shape.
    Returns:
    numpy.ndarray: A 1D array containing values from `Y` corresponding to the last valid index in `c_sal`.
    """
    last_values = np.full(c_sal.shape[1], np.nan)  # Initialize with NaN
    for col in range(c_sal.shape[1]):  # Iterate over each column
        valid_indices = np.where(~np.isnan(c_sal[:, col]))[0]  # Get valid indices
        if valid_indices.size > 0:  # If there are valid values
            last_index = valid_indices[-1]  # Get the last index
            last_values[col] = Y[last_index, col]  # Get the corresponding Y value
    return last_values

def detect_threshold_exceedance(c_sal, X, threshold=2):
    """
    Find the first column where a value in `c_sal` exceeds the threshold (default: 2).
    Return only the corresponding values from `X[:, index_found]`.
    Parameters:
    c_sal (numpy.ndarray): A 2D array (41, 422) containing values.
    X (numpy.ndarray): A 2D array (41, 422) of the same shape.
    threshold (float): The threshold value to check against (default is 2).
    Returns:
    numpy.ndarray or None: A 1D array `X[:, index_found]` if found, otherwise None.
    """
    for col in range(c_sal.shape[1]):  # Iterate through columns left to right
        if np.any(c_sal[:, col] > threshold):  # Check if any value exceeds threshold
            all_points = len(c_sal[1,:])
            return X[:, col ] , col# Return the corresponding column from X
    
    return None  # If no value > 2 is found, return None

# Function to read the .dat file
def read_dat_file(filename):
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

def read_references_along_branches(filename):
    point_id = []
    along_branch_name_id = []
    
    with open(filename, 'r') as file:
        # Skip the header line, then read each line
        next(file)
        for line in file:
            # Remove whitespace and split by comma
            branch, point, name = line.strip().split(',')
            # Append to your lists
            point_id.append(point)
            along_branch_name_id.append(name)
    return point_id, along_branch_name_id    

# Function to read the measurement files
def read_measurement_file(filename):
    # data = pd.read_csv(filename, sep='\t', parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d::%H:%M:%S'))
    data = pd.read_csv(
    filename,  # Replace with your actual file path
    sep='\t',  # Specify tab as the delimiter
    parse_dates=['Date'],  # Specify the column to parse as date
    date_format='%Y-%m-%d::%H:%M:%S'  # Use the date_format argument for parsing
    )
    return data

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

class HydroData(NamedTuple):
    time: np.ndarray
    level: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    total_depth: np.ndarray
    water_level: np.ndarray
    u_velocity: np.ndarray
    v_velocity: np.ndarray
    wl_dates: np.ndarray
    wl_values: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    distance: np.ndarray

class TSData(NamedTuple):
    time: np.ndarray
    level: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    total_depth: np.ndarray
    salinity: np.ndarray
    temperature: np.ndarray
    ts_dates: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    distance: np.ndarray

def hydro_load_netcdf(dataset_hydro, time_units, time_calendar):
    ####### Read the NetCDF file
    # dataset_hydro = Dataset(f'{models}/{river_branch}_hydro_extracted.nc')
    
    # Convert time using the adjusted time units
    t = num2date(dataset_hydro.variables['time'][:], units=time_units, calendar=time_calendar)
    
    level = dataset_hydro.variables['depth'][:]
    lon = dataset_hydro.variables['longitude'][:]
    lat = dataset_hydro.variables['latitude'][:]
    td = dataset_hydro.variables['total_depth'][:]
    wl = dataset_hydro.variables['water_level'][:]
    uv = dataset_hydro.variables['u_velocity'][:]
    vv = dataset_hydro.variables['v_velocity'][:]
    
    # Define time for water levels from the model
    wl_dates = np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in t])
    wl_values = wl[:, -1]
    
    td = td * -1
    
    # Calculate distances from each node
    c_d = [haversine(lat[i], lon[i], lat[i + 1], lon[i + 1]) for i in range(len(lat) - 1)]
    c_dd = [i * 1000 for i in c_d]
    c_d = [0] + c_dd
    dist = np.cumsum(c_d)
    dist = max(dist) - dist
    
    # Plot water levels
    X, Y = np.meshgrid(dist, level * -1)
    # The Y must be changed based on the water_level and depth
    return HydroData(
        time=t,
        level=level,
        lon=lon,
        lat=lat,
        total_depth=td,
        water_level=wl,
        u_velocity=uv,
        v_velocity=vv,
        wl_dates=wl_dates,
        wl_values=wl_values,
        X=X,
        Y=Y,
        distance=dist
    )

def ts_load_netcdf(dataset_ts, time_units, time_calendar):
    # Load TS Dataset
    # dataset_ts = Dataset(f'{models}/{river_branch}_ts_extracted.nc')
    
    sal = dataset_ts.variables['salinity'][:]
    tem = dataset_ts.variables['temperature'][:]
    
    # Convert time using the adjusted time units
    t = num2date(dataset_ts.variables['time'][:], units=time_units, calendar=time_calendar)
    
    level = dataset_ts.variables['depth'][:]
    lon = dataset_ts.variables['longitude'][:]
    lat = dataset_ts.variables['latitude'][:]
    td = dataset_ts.variables['total_depth'][:]
    
    td = td * -1
    
    # Calculate distances from each node
    c_d = [haversine(lat[i], lon[i], lat[i + 1], lon[i + 1]) for i in range(len(lat) - 1)]
    c_dd = [i * 1000 for i in c_d]
    c_d = [0] + c_dd
    dist = np.cumsum(c_d)
    dist = np.flipud(dist)
    
    # Plot water levels
    X, Y = np.meshgrid(dist, level * -1)
    
    # Create time array for TS data
    ts_dates = np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in t])
    
    return TSData(
        time=t,
        level=level,
        lon=lon,
        lat=lat,
        total_depth=td,
        salinity=sal,
        temperature=tem,
        ts_dates=ts_dates,
        X=X,
        Y=Y,
        distance=dist
    )
# %%

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define the two simulation paths
sim1_path = '/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/NC_out/deltapo_ER_202207_ogridNoDRCVmin'
sim2_path = '/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/NC_out/deltapo_ER_202207_ogridNoDRCVmin_WATEXT_f'

sim1_name = os.path.basename(sim1_path)
sim2_name = os.path.basename(sim2_path)

# Read discharge and water level boundary data
discharge_dates, discharge_values = read_dat_file(
    '/home/utente/Documenti/climaxpo/discharge_po_pontelagoscuro.dat')

# Add the other discharge branches
branch_data = {
    'PoDiPila': {'Qmed': -38.78, 'Qmax': -38.95},
    'PoDiGoro': {'Qmed': -1.17, 'Qmax': -1.19},
    'PoDiGnocca': {'Qmed': -9.04, 'Qmax': -9.22},
    'PoDiTolle': {'Qmed': -4.84, 'Qmax': -5.41},
    'PoDiMaistra': {'Qmed': -0.40, 'Qmax': -0.51}
}
total_qmax = sum(data['Qmax'] for data in branch_data.values())

# Choose what to plot
plot_salinity_diff = 1
plot_temperature_diff = 1
plot_velocity_diff = 1
plot_waterlevel_diff = 1
make_video = 1

# River branch
river_branch = 'PoDiVenezia'
# river_branch = 'PoDiGoro'

# Dates of interest
dt_fig = ["2022-07-01", "2022-09-01"]

# Plot limits
x_fig_lim = 50000
max_discharge_lim = 300
x_lim = [0, x_fig_lim]
y_lim = [-9.5, 1]

# Time settings
time_units = 'hours since 2022-07-01 00:00:00'
time_calendar = 'standard'

# Create output folder
output_folder = f'Diff_{sim1_name}_vs_{sim2_name}_{river_branch}'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(f'{output_folder}/plots', exist_ok=True)

# ============================================================================
# LOAD DATA FROM BOTH SIMULATIONS
# ============================================================================

print("Loading simulation 1 data...")
ds_hydro_1 = Dataset(f'{sim1_path}/{river_branch}_hydro_extracted.nc')
hydro_1 = hydro_load_netcdf(ds_hydro_1, time_units, time_calendar)

ds_ts_1 = Dataset(f'{sim1_path}/{river_branch}_ts_extracted.nc')
ts_1 = ts_load_netcdf(ds_ts_1, time_units, time_calendar)

print("Loading simulation 2 data...")
ds_hydro_2 = Dataset(f'{sim2_path}/{river_branch}_hydro_extracted.nc')
hydro_2 = hydro_load_netcdf(ds_hydro_2, time_units, time_calendar)

ds_ts_2 = Dataset(f'{sim2_path}/{river_branch}_ts_extracted.nc')
ts_2 = ts_load_netcdf(ds_ts_2, time_units, time_calendar)

# Verify both simulations have the same dimensions
if not np.array_equal(ts_1.time, ts_2.time):
    print("Warning: Time arrays don't match!")
    
# Use the X grid from first simulation (should be same for both)
X = hydro_1.X
t = ts_1.time

# Fix dates for plotting
dt_fig = pd.to_datetime(dt_fig)
start_date = dt_fig[0]
end_date = dt_fig[1]

# Contour levels for difference plots
diff_contour_levels_sal = np.arange(-5, 5.1, 1)
diff_contour_levels_temp = np.arange(-1, 1.1, 0.1)
diff_contour_levels_vel = np.arange(-0.25, 0.251, 0.05)

image_paths = []

# ============================================================================
# MAIN LOOP - CREATE DIFFERENCE PLOTS
# ============================================================================
# added -1 to match the times
for ii in range(1, len(t)-1, 1):
    current_time = t[ii]
    discharge_index = np.searchsorted(discharge_dates, current_time)
    
    # ------------------------------------------------------------------------
    # Extract data from both simulations
    # ------------------------------------------------------------------------
    
    # Temperature and salinity
    sal_1 = np.transpose(ts_1.salinity[ii, :, :])
    sal_2 = np.transpose(ts_2.salinity[ii, :, :])
    temp_1 = np.transpose(ts_1.temperature[ii, :, :])
    temp_2 = np.transpose(ts_2.temperature[ii, :, :])
    
    # Mask zeros
    sal_1[sal_1 == 0] = np.nan
    sal_2[sal_2 == 0] = np.nan
    temp_1[temp_1 == 0] = np.nan
    temp_2[temp_2 == 0] = np.nan
    
    # Hydrodynamic data
    uv_1 = np.transpose(hydro_1.u_velocity[ii, :, :])
    uv_2 = np.transpose(hydro_2.u_velocity[ii, :, :])
    vv_1 = np.transpose(hydro_1.v_velocity[ii, :, :])
    vv_2 = np.transpose(hydro_2.v_velocity[ii, :, :])
    
    uv_1[uv_1 == 0] = np.nan
    uv_2[uv_2 == 0] = np.nan
    vv_1[vv_1 == 0] = np.nan
    vv_2[vv_2 == 0] = np.nan
    
    # Velocity magnitude
    mag_1 = np.sqrt(uv_1**2 + vv_1**2)
    mag_2 = np.sqrt(uv_2**2 + vv_2**2)
    
    # Water level
    wl_1 = hydro_1.water_level[ii, :]
    wl_2 = hydro_2.water_level[ii, :]
    
    # ------------------------------------------------------------------------
    # Calculate differences (sim1 - sim2)
    # ------------------------------------------------------------------------
    sal_diff = sal_1 - sal_2
    temp_diff = temp_1 - temp_2
    uv_diff = uv_1 - uv_2
    mag_diff = mag_1 - mag_2
    wl_diff = wl_1 - wl_2
    
    # ------------------------------------------------------------------------
    # Setup Y coordinate
    # ------------------------------------------------------------------------
    # Get water level from hydro_1 for bathymetry
    wl_current = hydro_1.water_level[ii, :]
    
    # Calculate middle of each layer for Y1
    td = hydro_1.total_depth
    td = td.reshape(1, -1)
    Y1 = np.vstack((wl_current, hydro_1.Y[:, :]))
    middle_layers = (Y1[:-1, :] + Y1[1:, :]) / 2
    Y1_hydro = np.vstack((wl_current, middle_layers[:-1, :]))
    
    # ------------------------------------------------------------------------
    # Create figure with appropriate subplots
    # ------------------------------------------------------------------------
    n_plots = plot_salinity_diff + plot_temperature_diff + plot_velocity_diff
    if n_plots == 0:
        print("No plots selected!")
        break
    
    fig = plt.figure(figsize=(30, 15))
    fig.set_size_inches(30, 15, forward=True)  # Force exact size
    
    if n_plots == 1:
        gs = gridspec.GridSpec(8, 1)
        ax1 = fig.add_subplot(gs[0:7, :])
        ax2 = fig.add_subplot(gs[7:8, :])  # Discharge plot
    elif n_plots == 2:
        gs = gridspec.GridSpec(8, 1)
        ax1 = fig.add_subplot(gs[0:3, :])  # First variable
        ax2 = fig.add_subplot(gs[3:6, :])  # Second variable
        ax3 = fig.add_subplot(gs[6:8, :])  # Discharge plot (taller)
    else:  # 3 plots
        gs = gridspec.GridSpec(9, 1)
        ax1 = fig.add_subplot(gs[0:3, :])  # Salinity diff
        ax2 = fig.add_subplot(gs[3:6, :])  # Temperature diff
        ax3 = fig.add_subplot(gs[6:8, :])  # Velocity diff
        ax4 = fig.add_subplot(gs[8:9, :])  # Discharge plot
    
    # ------------------------------------------------------------------------
    # Plot differences
    # ------------------------------------------------------------------------
    plot_idx = 0
    
    # Mask for NaN regions
    mask = np.ones_like(sal_diff)
    mask[np.isnan(sal_diff)] = 0
    
    # Salinity difference
    if plot_salinity_diff:
        if n_plots == 1:
            current_ax = ax1
        elif n_plots == 2:
            current_ax = ax1 if plot_idx == 0 else ax2
        else:
            current_ax = ax1
            
        surf_s = current_ax.contourf(X, Y1_hydro, sal_diff, 
                                      cmap='RdBu_r', 
                                      levels=diff_contour_levels_sal,
                                      extend='both',
                                      zorder=2)
        current_ax.contourf(X, Y1_hydro, mask, colors='gray', 
                            levels=[-0.5, 0.5, 1.5], alpha=1.0, zorder=1)
        cbar = plt.colorbar(surf_s, ax=current_ax, fraction=0.1, pad=0.04)
        cbar.set_label('Salinity Difference (PSU)', fontsize=14)
        current_ax.set_title(f'Salinity: {sim1_name} - {sim2_name}', fontsize=16)
        current_ax.set_xlim(x_lim)
        current_ax.set_ylim(y_lim)
        plot_idx += 1
    
    # Temperature difference
    if plot_temperature_diff:
        if n_plots == 1:
            current_ax = ax1
        elif n_plots == 2:
            current_ax = ax1 if plot_idx == 0 else ax2
        else:
            current_ax = ax2
            
        surf_t = current_ax.contourf(X, Y1_hydro, temp_diff,
                                      cmap='RdBu_r',
                                      levels=diff_contour_levels_temp,
                                      extend='both',
                                      zorder=2)
        current_ax.contourf(X, Y1_hydro, mask, colors='gray',
                            levels=[-0.5, 0.5, 1.5], alpha=1.0, zorder=1)
        cbar = plt.colorbar(surf_t, ax=current_ax, fraction=0.1, pad=0.04)
        cbar.set_label('Temperature Difference (°C)', fontsize=14)
        current_ax.set_title(f'Temperature: {sim1_name} - {sim2_name}', fontsize=16)
        plot_idx += 1
    
    # Velocity magnitude difference
    if plot_velocity_diff:
        if n_plots == 1:
            current_ax = ax1
        elif n_plots == 2:
            current_ax = ax1 if plot_idx == 0 else ax2
        else:
            current_ax = ax3
            
        surf_v = current_ax.contourf(X, Y1_hydro, mag_diff,
                                      cmap='RdBu_r',
                                      levels=diff_contour_levels_vel,
                                      extend='both',
                                      zorder=2)
        current_ax.contourf(X, Y1_hydro, mask, colors='gray',
                            levels=[-0.5, 0.5, 1.5], alpha=1.0, zorder=1)
        cbar = plt.colorbar(surf_v, ax=current_ax, fraction=0.1, pad=0.04)
        cbar.set_label('Velocity Difference (m/s)', fontsize=14)
        current_ax.set_title(f'Velocity Magnitude: {sim1_name} - {sim2_name}', fontsize=16)
        plot_idx += 1
    
    # ------------------------------------------------------------------------
    # Add common elements to all upper plots
    # ------------------------------------------------------------------------
    upper_axes = []
    if plot_salinity_diff:
        upper_axes.append(ax1 if n_plots > 1 else ax1)
    if plot_temperature_diff:
        upper_axes.append(ax2 if n_plots > 2 else (ax1 if n_plots == 1 else ax2))
    if plot_velocity_diff and n_plots > 2:
        upper_axes.append(ax3)
    
    for ax in upper_axes:
        # Plot water levels from both simulations
        ax.plot(X[0, :], wl_1, 'k-', linewidth=2, label=f'{sim1_name} WL')
        ax.plot(X[0, :], wl_2, 'r--', linewidth=2, label=f'{sim2_name} WL')
        
        # Plot bathymetry
        ax.fill_between(X[0, :], -td.flatten(), wl_1, color='lightgray', alpha=0.3)
        
        # Formatting
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_ylabel('Depth (m)', fontsize=18)
        ax.tick_params(labelsize=16)
        ax.invert_xaxis()
        ax.legend(loc='upper right', fontsize=12)
    
    # ------------------------------------------------------------------------
    # Water level difference plot (if selected)
    # ------------------------------------------------------------------------
    if plot_waterlevel_diff:
        # Create a small inset or additional subplot for water level difference
        if n_plots <= 2:
            # Add an extra axis on the right side
            for ax in upper_axes:
                # Create twin axis for water level difference
                ax2_twin = ax.twinx()
                ax2_twin.plot(X[0, :], wl_diff, 'g-', linewidth=2, label='WL Diff')
                ax2_twin.set_ylabel('WL Difference (m)', color='g', fontsize=14)
                ax2_twin.tick_params(axis='y', labelcolor='g')
                ax2_twin.legend(loc='lower right', fontsize=12)
    
    # ------------------------------------------------------------------------
    # Discharge plot (bottom)
    # ------------------------------------------------------------------------
    if n_plots == 1:
        discharge_ax = ax2
    elif n_plots == 2:
        discharge_ax = ax3
    else:
        discharge_ax = ax4
    
    discharge_ax.plot(discharge_dates, discharge_values, 'k', linewidth=2)
    if 0 <= discharge_index < len(discharge_dates):
        current_value = discharge_values[discharge_index]
        current_date = discharge_dates[discharge_index]
        
        discharge_ax.plot(current_date, current_value, 'ro', markersize=8)
        discharge_ax.annotate(f'{current_value:.1f} m³/s',
                              xy=(current_date, current_value),
                              xytext=(10, 10),
                              textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->'))
    
    discharge_ax.set_xlim([start_date, end_date])
    discharge_ax.set_ylim([0, max_discharge_lim])
    discharge_ax.set_ylabel('Discharge (m³/s)', fontsize=16)
    discharge_ax.grid(True, alpha=0.3)
    
    # Add branch Qmax lines
    for branch_name, data in branch_data.items():
        discharge_ax.axhline(y=abs(data['Qmax']), linestyle='--', alpha=0.5,
                             label=f"{branch_name}: {abs(data['Qmax']):.2f}")
    
    discharge_ax.axhline(y=abs(total_qmax), linestyle='--', linewidth=2, 
                         color='k', label=f"Total Qmax: {abs(total_qmax):.2f}")
    
    # Adjust legend position
    pos = discharge_ax.get_position()
    discharge_ax.legend(bbox_to_anchor=(0.90, 1.40), loc='upper left', fontsize=10)
    
    # ------------------------------------------------------------------------
    # Final formatting
    # ------------------------------------------------------------------------
    # Add main title
    fig.suptitle(f'Difference: {sim1_name} vs {sim2_name}, Time: {current_time.strftime("%Y-%b-%d %H:%M")} - Branch: {river_branch}',
                 fontsize=20, y=0.98)
    
    # Set x-label on bottom plot
    if n_plots == 1:
        discharge_ax.set_xlabel('Distance along river (m)', fontsize=18, labelpad=15)
    elif n_plots == 2:
        discharge_ax.set_xlabel('Distance along river (m)', fontsize=18, labelpad=15)
    else:
        ax3.set_xlabel('Distance along river (m)', fontsize=18, labelpad=15)
        discharge_ax.set_xlabel('')  # Remove duplicate
    
    plt.tight_layout()
    # plt.tight_layout(pad=2.0)  # Use consistent padding
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    figure_name = os.path.join(output_folder, 'plots', f'diff_plot_{ii:04d}.png')
    plt.savefig(figure_name, dpi=150, bbox_inches='tight')
    plt.close()
    
    image_paths.append(figure_name)
    
    if ii % 10 == 0:
        print(f"Processed {ii}/{len(t)-1} time steps")

# ============================================================================
# CREATE VIDEO
# ============================================================================
if make_video and image_paths:
    output_video = os.path.join(output_folder, f'{river_branch}_difference.mp4')
    with imageio.get_writer(output_video, mode='I', fps=4) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    print(f"Video saved: {output_video}")

print("Done!")