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

# Read discharge and water level boundary data
discharge_dates, discharge_values = read_dat_file(
    '/home/utente/Documenti/climaxpo/discharge_po_pontelagoscuro.dat')

# Define folder where NC files are
models = '/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/NC_out/deltapo_ER_202207_ogridNoDRCVmin'
# models = '/home/utente/Documenti/climaxpo/po-er_hindcast_202207/po-er_hindcast/sims/202207/NC_out/deltapo_ER_202207_ogridNoDRCVmin_WATEXT_f'
sims_name = os.path.basename(models)
# Check if hydro, ts or measurements will be plotted, and if video will be save
# There will be 2 plots based in the inputs from here, if hydro=1 there will be currents and water levels, if ts=1 there will be salinity and temperature
hydro = 0
ts = 1 
measurements = 0
make_video = 1

# Based in previous inputs, 1 figure with 3 plots can be done based on this,
# hydro=1 && ts=0, there will be the water levels and velocity magnitude, top=[wl+vel_mag+quiver of velocity]; middle=[wl+u_vel]; lower=[discharge from the po]
# hydro=0 && ts=1, there will be only temperature and salinity without water levels, top=[salinity]; middle=[temperature]; lower=[discharge from the po]
# hydro=1 && ts=1, there will be the water levels and velocity magnitude, and salinity and temperature, top=[wl+quiver of velocity+salinity]; middle=[wl+quivers+temperature]; lower=[discharge from the po]

# Determine suffix based on flags
if hydro == 1 and ts == 1:
    suffix = '_hydro_ts'
elif hydro == 1 and ts == 0:
    suffix = '_hydro'
elif hydro == 0 and ts == 1:
    suffix = '_ts'
else:
    suffix = ''

# dates of interest to be plotted
dt_fig = ["2022-07-01", "2022-09-01"]
store_swi = np.empty((0,4)) # First column is date, second is length in km, third is lat, fourth is lon
river_branch = 'PoDiVenezia'

# Max length of the river to be plotted (both for ax1,ax2), and of the Y axis in the discharge plot ax3
x_fig_lim = 50000
max_discharge_lim = 750
x_lim = [0,x_fig_lim]
y_lim = [-9.5, 1]

# Set time for the NC files
time_units = 'hours since 2022-07-15 00:00:00'
time_calendar = 'standard'

# Load the NETCDF files for hydro and t/s based on flags
if hydro == 1:
    dataset_hydro = Dataset(f'{models}/{river_branch}_hydro_extracted.nc')
    hydro_ds = hydro_load_netcdf(dataset_hydro, time_units, time_calendar)
    
if ts == 1: 
    dataset_ts = Dataset(f'{models}/{river_branch}_ts_extracted.nc')
    ts_ds = ts_load_netcdf(dataset_ts, time_units, time_calendar)

# Setting global variables for vertical branch plot
if hydro == 1:
    X = hydro_ds.X
    t = hydro_ds.time
    # We'll handle Y1 dynamically in the loop based on the case
elif ts == 1:
    X = ts_ds.X
    t = ts_ds.time
else:
    print("Error: Both hydro and ts cannot be 0")
    exit()

if measurements == 1:
    # Measurement points and files
    dist_points = [48591, 48499, 46915, 44004, 41644, 38871, 35719]
    files = ['202', '203', '204', '205', '206', '207']
    # Read measurement data
    measurement_data = {file_id: read_measurement_file(
        f'C:\\Users\\a_p_h\\OneDrive\\CLIMAXPO\\data\\2003_2017\\Salt_Wedge_Measurements_Turolla\\2017\\output\\to_shyfem_interpolated_2017_06_28_data\\measured_salt_{file_id}_corrected.txt') for file_id in files}

# Create folder to save new images
folder_name = f'{models}_{river_branch}_plots'
folder_name_plots = f'{folder_name}/vertical_plots'
os.makedirs(folder_name, exist_ok=True)
os.makedirs(folder_name_plots, exist_ok=True)

# Save plots as images and create a list to store image paths
image_paths = []

# Fix dates to show in plot
dt_fig = pd.to_datetime(dt_fig)
start_date = dt_fig[0]
end_date = dt_fig[1]

# Set contour levels for plots
contour_levels_s = np.array([0,2,5,10,15,20,25,30,35,40])
contour_levels_t = np.array([16,18,20,22,24,26,28,30,32])
# %%

# Save plots as images
for ii in range(1, len(t), 1):
    c_t = t[ii]
    discharge_index = np.searchsorted(discharge_dates, c_t)
    
    # Initialize variables based on available data
    if hydro == 1:
        # Find the indices in wl_dates and discharge_dates that are closest to closest time (c_t)
        wl_index = np.searchsorted(hydro_ds.wl_dates, c_t)
        # Extract current velocity matrix
        c_uv = np.transpose(hydro_ds.u_velocity[ii,:,:])
        c_vv = np.transpose(hydro_ds.v_velocity[ii,:,:])
        c_uv[c_uv == 0] = np.nan
        c_vv[c_vv == 0] = np.nan
        c_m = np.sqrt(c_uv**2 + c_vv**2)
        
    if ts == 1:
        # Extract current salinity and temperature
        c_sal = np.transpose(ts_ds.salinity[ii,:,:])
        c_tem = np.transpose(ts_ds.temperature[ii,:,:])
        c_sal[c_sal == 0] = np.nan
        c_tem[c_tem == 0] = np.nan
        
        # For SWI detection (only relevant when ts is on)
        if ts == 1:
            X_length, col_X = detect_threshold_exceedance(c_sal, X)
    
    # Calculate Y1 based on the case
    if hydro == 1:
        # Calculate middle of each layer for hydro
        td = hydro_ds.total_depth
        td = td.reshape(1,-1)
        Y1 = np.vstack((hydro_ds.water_level[ii, :], hydro_ds.Y[:, :]))  # add current WL to layer
        middle_layers = (Y1[:-1, :] + Y1[1:, :]) / 2  # get mean of layers
        Y1_hydro = np.vstack((hydro_ds.water_level[ii, :], middle_layers[:-1, :]))  # get matrix for surf plot
        
        if ts == 1:
            # Case: both hydro and ts
            Y1 = Y1_hydro
        else:
            # Case: hydro only
            Y1 = Y1_hydro
    else:
        # Case: ts only
        Y_ts = ts_ds.Y[:,:] + 0.25
        # top_zeroes = np.zeros(len(Y_ts))
        # Y1_ts = np.vstack((top_zeroes, ts_ds.Y[:, :]))  # get matrix for surf plot
        Y1 = Y_ts
    
    # Create figure with appropriate GridSpec
    fig = plt.figure(figsize=(30,15))
    
    # Create GridSpec with 9 rows
    gs = gridspec.GridSpec(9, 1)
    
    # Determine which subplots to use based on hydro/ts flags
    if hydro == 1 and ts == 0:
        # CASE 1: hydro=1, ts=0
        # top=[wl+vel_mag+quiver of velocity]; middle=[wl+u_vel]; lower=[discharge]
        ax1 = fig.add_subplot(gs[0:4,:])  # Top plot: velocity magnitude
        ax2 = fig.add_subplot(gs[4:8,:])  # Middle plot: u_velocity
        ax3 = fig.add_subplot(gs[8:9,:])  # Bottom plot: discharge
        
        # Top plot: velocity magnitude
        surf_mag = ax1.contourf(X, Y1, c_m, cmap='viridis', levels=20, zorder=2)
        plt.colorbar(surf_mag, ax=ax1, label='Velocity Magnitude (m/s)')
        
        # Middle plot: u_velocity
        surf_u = ax2.contourf(X, Y1, c_uv, cmap='RdBu_r', levels=20, zorder=2)
        plt.colorbar(surf_u, ax=ax2, label='U Velocity (m/s)')
        
        # Plot water level on both top plots
        ax1.plot(X[0,:], hydro_ds.water_level[ii, :], 'k', linewidth=4)
        ax2.plot(X[0,:], hydro_ds.water_level[ii, :], 'k', linewidth=4)
        
        # Plot bathymetry
        td = hydro_ds.total_depth
        ax1.fill_between(X[0,:], -td, hydro_ds.water_level[ii,:], color='lightgray', alpha=0.5)
        ax2.fill_between(X[0,:], -td, hydro_ds.water_level[ii,:], color='lightgray', alpha=0.5)
        
        # Add quivers for flow direction (only on top plot)
        n = 3  # Adjust as needed for Y direction
        m = 10  # Adjust as needed for X direction
        
        # Normalize vectors
        magnitude = np.sqrt(c_uv**2 + c_vv**2)
        un = c_uv / magnitude
        vn = c_vv / magnitude
        un = np.nan_to_num(un)
        vn = np.nan_to_num(vn)
        
        u_direction = np.sign(c_uv[0::n, 0::m])
        v_direction = np.zeros_like(u_direction)
        
        mask_right = u_direction > 0
        mask_left = u_direction < 0
        
        # Plot quivers
        ax1.quiver(X[0::n, 0::m][mask_right], Y1[0::n, 0::m][mask_right],
                   u_direction[mask_right], v_direction[mask_right],
                   scale=6, scale_units='inches', width=0.002, headwidth=4, 
                   color='black', label='Out', zorder=3)
        
        ax1.quiver(X[0::n, 0::m][mask_left], Y1[0::n, 0::m][mask_left],
                   u_direction[mask_left], v_direction[mask_left],
                   scale=6, scale_units='inches', width=0.002, headwidth=4, 
                   color='blue', label='In', zorder=3)
        
    elif hydro == 0 and ts == 1:
        # CASE 2: hydro=0, ts=1
        # top=[salinity]; middle=[temperature]; lower=[discharge]
        ax1 = fig.add_subplot(gs[0:4,:])  # Top plot: salinity
        ax2 = fig.add_subplot(gs[4:8,:])  # Middle plot: temperature
        ax3 = fig.add_subplot(gs[8:9,:])  # Bottom plot: discharge
        
        # Salinity plot
        surf_s = ax1.contourf(X, Y1, c_sal, cmap=cmo.haline, levels=contour_levels_s, 
                              vmin=0, vmax=40, zorder=2)
        # Mask NaN regions
        mask = np.ones_like(c_sal)
        mask[np.isnan(c_sal)] = 0
        ax1.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5], 
                     alpha=1.0, zorder=1)
        cbar_s = plt.colorbar(surf_s, ax=ax1, label='Salinity (PSU)')
        
        # Temperature plot
        surf_t = ax2.contourf(X, Y1, c_tem, cmap=cmo.thermal, levels=contour_levels_t,
                              vmin=16, vmax=32, zorder=2)
        ax2.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5],
                     alpha=1.0, zorder=1)
        cbar_t = plt.colorbar(surf_t, ax=ax2, label='Temperature (°C)')
        
        # Add contour lines
        contour_colors_ax1 = ['w', 'w', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
        contours_s = ax1.contour(X, Y1, c_sal, colors=contour_colors_ax1, 
                                  levels=contour_levels_s, linewidths=2)
        contours_t = ax2.contour(X, Y1, c_tem, colors=contour_colors_ax1,
                                  levels=contour_levels_t, linewidths=2)
        
        # SWI detection for salinity plot
        if 'X_length' in locals() and X_length is not None and col_X is not None:
            x_value = X[:, col_X][0]
            ax1.axvline(x=x_value, color='red', linestyle='--', linewidth=2, 
                        label=f"Threshold @ {x_value:.2f}")
            x_value_km = x_value / 1000
            new_entry = np.array([[ts_ds.time[ii], x_value_km, 
                                  np.flip(ts_ds.lon[col_X]), np.flip(ts_ds.lat[col_X])]])
            store_swi = np.vstack((store_swi, new_entry))
            ax1.text(x_value + 0.2, Y1.max(), f"{x_value_km:.2f} km", fontsize=12, 
                     color='black', bbox=dict(facecolor='white', edgecolor='black', 
                                             boxstyle='round,pad=0.3'))
        
    elif hydro == 1 and ts == 1:
        # CASE 3: hydro=1, ts=1
        # top=[wl+quiver of velocity+salinity]; middle=[wl+quivers+temperature]; lower=[discharge]
        ax1 = fig.add_subplot(gs[0:4,:])  # Top plot: salinity with quivers
        ax2 = fig.add_subplot(gs[4:8,:])  # Middle plot: temperature with quivers
        ax3 = fig.add_subplot(gs[8:9,:])  # Bottom plot: discharge
        
        # Salinity plot (top)
        surf_s = ax1.contourf(X, Y1, c_sal, cmap=cmo.haline, levels=contour_levels_s,
                              vmin=0, vmax=40, zorder=2)
        # Mask NaN regions
        mask = np.ones_like(c_sal)
        mask[np.isnan(c_sal)] = 0
        ax1.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5],
                     alpha=1.0, zorder=1)
        cbar_s = plt.colorbar(surf_s, ax=ax1, label='Salinity (PSU)')
        
        # Temperature plot (middle)
        surf_t = ax2.contourf(X, Y1, c_tem, cmap=cmo.thermal, levels=contour_levels_t,
                              vmin=16, vmax=32, zorder=2)
        ax2.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5],
                     alpha=1.0, zorder=1)
        cbar_t = plt.colorbar(surf_t, ax=ax2, label='Temperature (°C)')
        
        # Add contour lines
        contour_colors_ax1 = ['w', 'w', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
        contours_s = ax1.contour(X, Y1, c_sal, colors=contour_colors_ax1,
                                  levels=contour_levels_s, linewidths=2)
        contours_t = ax2.contour(X, Y1, c_tem, colors=contour_colors_ax1,
                                  levels=contour_levels_t, linewidths=2)
        
        # Plot water level on both plots
        ax1.plot(X[0,:], hydro_ds.water_level[ii, :], 'k', linewidth=4)
        ax2.plot(X[0,:], hydro_ds.water_level[ii, :], 'k', linewidth=4)
        
        # Plot bathymetry
        td = hydro_ds.total_depth
        ax1.fill_between(X[0,:], -td, hydro_ds.water_level[ii,:], color='lightgray', alpha=0.3)
        ax2.fill_between(X[0,:], -td, hydro_ds.water_level[ii,:], color='lightgray', alpha=0.3)
        
        # Add quivers for flow direction on both plots
        n = 3
        m = 10
        
        magnitude = np.sqrt(c_uv**2 + c_vv**2)
        un = c_uv / magnitude
        vn = c_vv / magnitude
        un = np.nan_to_num(un)
        vn = np.nan_to_num(vn)
        
        u_direction = np.sign(c_uv[0::n, 0::m])
        v_direction = np.zeros_like(u_direction)
        
        mask_right = u_direction > 0
        mask_left = u_direction < 0
        
        # Quivers on top plot
        ax1.quiver(X[0::n, 0::m][mask_right], Y1[0::n, 0::m][mask_right],
                   u_direction[mask_right], v_direction[mask_right],
                   scale=6, scale_units='inches', width=0.002, headwidth=4,
                   color='black', label='Out', zorder=3)
        ax1.quiver(X[0::n, 0::m][mask_left], Y1[0::n, 0::m][mask_left],
                   u_direction[mask_left], v_direction[mask_left],
                   scale=6, scale_units='inches', width=0.002, headwidth=4,
                   color='blue', label='In', zorder=3)
        
        # Quivers on middle plot
        ax2.quiver(X[0::n, 0::m][mask_right], Y1[0::n, 0::m][mask_right],
                   u_direction[mask_right], v_direction[mask_right],
                   scale=6, scale_units='inches', width=0.002, headwidth=4,
                   color='black', zorder=3)
        ax2.quiver(X[0::n, 0::m][mask_left], Y1[0::n, 0::m][mask_left],
                   u_direction[mask_left], v_direction[mask_left],
                   scale=6, scale_units='inches', width=0.002, headwidth=4,
                   color='blue', zorder=3)
        
        # SWI detection for salinity plot
        if 'X_length' in locals() and X_length is not None and col_X is not None:
            x_value = X[:, col_X][0]
            ax1.axvline(x=x_value, color='red', linestyle='--', linewidth=2,
                        label=f"Threshold @ {x_value:.2f}")
            x_value_km = x_value / 1000
            new_entry = np.array([[ts_ds.time[ii], x_value_km,
                                  np.flip(ts_ds.lon[col_X]), np.flip(ts_ds.lat[col_X])]])
            store_swi = np.vstack((store_swi, new_entry))
            ax1.text(x_value + 0.2, Y1.max(), f"{x_value_km:.2f} km", fontsize=12,
                     color='black', bbox=dict(facecolor='white', edgecolor='black',
                                             boxstyle='round,pad=0.3'))
    
    else:
        print("Error: Invalid combination of hydro and ts flags")
        exit()
    
    # Add measurements if enabled (only for salinity/temperature plots)
    if measurements == 1 and ts == 1:
        for file_id, dist_point in zip(files, dist_points):
            data = measurement_data[file_id]
            c_dist = np.mean(data['Distance'])
            
            if hydro == 1:
                # Find the closest distance in the dist array for hydro case
                closest_idx = np.abs(hydro_ds.dist - c_dist).argmin()
                closest_wl = hydro_ds.water_level[ii, closest_idx]
            else:
                # For ts only case, use ts_ds
                closest_idx = np.abs(ts_ds.distance - c_dist).argmin()
                closest_wl = 0  # No water level in ts only case
            
            # Insert new depth level at the water level
            new_depth = pd.DataFrame({
                'Distance': [data['Distance'].iloc[0]],
                'Depth': [-closest_wl],
                'Salinity': [data['Salinity'].iloc[0]],
                'Date': [data['Date'].iloc[0]]
            })
            data = pd.concat([new_depth, data], ignore_index=True)
            
            ax1.scatter(data['Distance'], data['Depth'] * -1, c=data['Salinity'],
                        cmap='turbo', vmin=1, vmax=35, edgecolors='w', s=10, zorder=5)
    
    # Add max SWI line if we have stored data
    if store_swi.size > 0 and ts == 1:
        max_swi_length = np.max(store_swi[:, 1])
        max_indices = np.where(store_swi[:, 1] == max_swi_length)[0]
        max_index = max_indices[-1]
        max_time, max_swi_length, max_lon, max_lat = store_swi[max_index]
        max_time_str = str(max_time)
        
        ax1.axvline(x=max_swi_length * 1000, color='magenta', linestyle='--', linewidth=2,
                    label=f"Max SWI @ {max_swi_length:.2f} km")
        ax1.text(max_swi_length * 1000 + 0.2, Y1.max() - 2,
                 f"Max Length: {max_swi_length:.2f} km\n{max_time_str}",
                 fontsize=12, color='black',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    # Common axis formatting for ax1 and ax2
    xticks = np.arange(0, 50001, 5000)
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xticks(xticks)
        ax.set_ylabel('Depth (m)', fontsize=18)
        ax.tick_params(labelsize=20)
        ax.invert_xaxis()
    
    # Specific formatting
    ax1.set_title(f'Time: {c_t.strftime("%Y-%b-%d %H:%M")} model used -> {sims_name} -> Branch: {river_branch}', 
                  fontsize=22)
    ax2.set_xlabel('Distance along river (m)', fontsize=18, labelpad=15)
    
    # Discharge plot formatting
    ax3.plot(discharge_dates, discharge_values, 'k')
    if 0 <= discharge_index < len(discharge_dates):
        ax3.plot(discharge_dates[discharge_index], discharge_values[discharge_index], 'ro')
    ax3.set_xlim([start_date, end_date])
    ax3.set_ylim([0, max_discharge_lim])
    ax3.set_ylabel('Discharge (m³/s)')
    ax3.grid(True)
    
    fig.tight_layout()
    
    # figure_name = os.path.join(folder_name_plots, f'plot_{ii:04d}.png')
    figure_name = os.path.join(folder_name_plots, f'plot_{ii:04d}{suffix}.png')
    print(f"Saving: {figure_name}")
    plt.savefig(figure_name)
    plt.close()
    
    image_paths.append(figure_name)

# Save stored data from the max salinity
if store_swi.size > 0:
    f_store_swi = folder_name + f'/results_{river_branch}_stored_swi.txt'
    
    with open(f_store_swi, 'w') as f:
        for row in store_swi:
            date_str = row[0].isoformat(sep='::')
            length_km = f"{row[1]:.4f}"
            lat = f"{row[2]:.8f}"
            lon = f"{row[3]:.8f}"
            f.write(f"{date_str}\t{length_km}\t{lat}\t{lon}\n")

# Create a video from the saved images
if make_video == 1:
    # output_video = os.path.join(folder_name, f'{river_branch}_video4.mp4')
    output_video = os.path.join(folder_name, f'{river_branch}{suffix}_video4.mp4')
    with imageio.get_writer(output_video, mode='I', fps=4) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    print("Video saved at", output_video)