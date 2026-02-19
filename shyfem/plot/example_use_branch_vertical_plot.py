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
hydro = 1
ts = 1 
measurements = 0
make_video = 1

dt_fig = ["2022-07-01", "2022-09-01"]# ["2022-06-01", "2022-07-01"] # Values considered for the plot (when discharge and water levels are shown)
store_swi = np.empty((0,4)) # First column is date, second is length in km, third is lat, fourth is lon
# The NC files of each branch contain lon,lat,level/depth,total_depth,time,salinity,temperature,water_level,u_vel,v_vel
# river_branch = 'PoDiVenezia' # ['PoDiVenezia','PoDiGoro','PoDiGnocca','PoDiTolle','PoDiTramontana']
river_branch = 'PoDiGoro' # ['PoDiVenezia','PoDiGoro','PoDiGnocca','PoDiTolle','PoDiTramontana']

x_fig_lim = 50000
max_discharge_lim = 750

x_lim = [0,x_fig_lim]  # x_lim = [np.min(X), np.max(X)] # 35000, 49000
y_lim = [-9.5, 1]

# Show names for better identificabilty
along_river_reference_filename = '/home/utente/Documenti/climaxpo/references_along_branches.csv'
along_river_references = read_references_along_branches(along_river_reference_filename)

# Set time for the NC files -> [check if this can be read for each NC file]
time_units = 'hours since 2022-07-15 00:00:00'
time_calendar = 'standard'  # Assuming standard calendar

# Load the NETCDF files for hydro and t/s
if hydro == 1:
    dataset_hydro = Dataset(f'{models}/{river_branch}_hydro_extracted.nc')
    hydro_ds = hydro_load_netcdf(dataset_hydro, time_units, time_calendar)
    
if ts == 1: 
    dataset_ts = Dataset(f'{models}/{river_branch}_ts_extracted.nc')
    ts_ds = ts_load_netcdf(dataset_ts, time_units, time_calendar)

# Setting global variables for vertical branch plot, because are repeated in each NC file.
if hydro == 1 and ts == 1:
    # grid
    X = hydro_ds.X
    Y = hydro_ds.Y
    t = hydro_ds.time
    
elif hydro == 1 and ts == 0:
    # grid
    X = hydro_ds.X
    Y = hydro_ds.Y
    t = hydro_ds.time
    
elif hydro == 0 and ts == 1:
    # grid
    X = hydro_ds.X
    Y = ts_ds.Y
    t = ts_ds.time
    
else:  # hydro == 0 and ts == 0
    # grid
    exit

if measurements == 1:
    # Measurement points and files
    dist_points = [48591, 48499, 46915, 44004, 41644, 38871, 35719]
    files = ['202', '203', '204', '205', '206', '207']
    # Read measurement data
    measurement_data = {file_id: read_measurement_file(
        f'C:\\Users\\a_p_h\\OneDrive\\CLIMAXPO\\data\\2003_2017\\Salt_Wedge_Measurements_Turolla\\2017\\output\\to_shyfem_interpolated_2017_06_28_data\\measured_salt_{file_id}_corrected.txt') for file_id in files}

# create folder to save new images
folder_name = f'{models}_{river_branch}_plots' # sim_folder + f'/results/output/{case_model}/{dates_NC}_{case_model}_{river_branch}_001'
folder_name_plots = f'{folder_name}/vertical_plots' # sim_folder + f'/results/output/{case_model}/{dates_NC}_{case_model}_{river_branch}_001/plots2'
os.makedirs(folder_name, exist_ok=True)
os.makedirs(folder_name_plots, exist_ok=True)

# Save plots as images and create a list to store image paths
image_paths = []

# Fix dates to show in plot
dt_fig = pd.to_datetime(dt_fig)
start_date = dt_fig[0]
end_date = dt_fig[1]
# Set contour levels for plot, this might change because of cmocean
contour_levels_s = np.array([0,2,5,10,15,20,25,30,35,40]) # np.arange(0, 37, 10)  # Contours at every 2 ppt from 0 to 36
contour_levels_t = np.array([16,18,20,22,24,26,28,30,32]) # np.arange(0, 37, 10)  # Contours at every 2 ppt from 0 to 36
# %%
        
# Save plots as images
for ii in range(1, len(t), 1 ):  # 1, len(t), 1 
    c_t = t[ii]
    # print(c_t, ' , ',ii)
    discharge_index = np.searchsorted(discharge_dates, c_t)
    
    if hydro == 1:
        # Find the indices in wl_dates and discharge_dates that are closest to closest time (c_t)
        wl_index = np.searchsorted(hydro_ds.wl_dates, c_t)
        # Extract current velocity matrix
        c_uv = np.transpose(hydro_ds.u_velocity[ii,:,:])
        c_vv = np.transpose(hydro_ds.v_velocity[ii,:,:])
        c_uv[c_uv == 0] = np.nan
        c_vv[c_vv == 0] = np.nan
        c_m = np.sqrt(c_uv**2 + c_vv**2)
        # Calculate middle of each layer
        td = hydro_ds.total_depth; td=td.reshape(1,-1)
        Y1 = np.vstack((hydro_ds.water_level[ii, :], hydro_ds.Y[:, :]))  # add current WL to layer
        middle_layers = (Y1[:-1, :] + Y1[1:, :]) / 2  # get mean of layers
        # When only hydro_ts the top row is change to the water levels, and is updated in each time step
        Y1_hydro = np.vstack((hydro_ds.water_level[ii, :], middle_layers[:-1, :]))  # get matrix for surf plot
        # This was for testing
        # surf = ax1.pcolormesh(X, Y1, c_sal, shading='gouraud', cmap='turbo', vmin=1, vmax=35)
        
    if ts == 1:
        # Extract current salinity 
        c_sal = np.transpose(ts_ds.salinity[ii,:,:])
        c_tem = np.transpose(ts_ds.temperature[ii,:,:])
        c_sal[c_sal == 0] = np.nan
        c_tem[c_tem == 0] = np.nan
        # c_sal[c_sal > 35] = 35
        # With no hydro data, we select last value of each column
        last_used_row = get_last_values(c_sal, ts_ds.Y)
        # Y1 = Y
        # middle_layers = (Y1[:-1, :] + Y1[1:, :]) / 2  # get mean of layers
        top_zeroes = np.zeros(len(Y[0,:]))
        Y1_ts = np.vstack((top_zeroes, middle_layers[:-1, :], Y[-1,:]))  # get matrix for surf plot
        X_length,col_X = detect_threshold_exceedance(c_sal, X)
        
    if hydro == 1 and ts == 1:
        Y1 = Y1_hydro
        # Check visually by plotting bottom and water level
        # td = hydro_ds.total_depth; wl_all = hydro_ds.water_level
        # plt.plot(X[0,:], td, 'k', linewidth=1); plt.grid(True); plt.xlim(-1000,50000)
        # plt.plot(X[0,:], wl_all[-1], 'b', linewidth=1)
    
    # fig, ax = plt.subplots(figsize=(30, 15))
    fig = plt.figure(figsize=(30,15))
    
    # Create a GridSpec with 4 rows and 4 columns
    gs = gridspec.GridSpec(9, 1)
    # Subplot that spans from position 1 to 8
    ax1 = fig.add_subplot(gs[0:4,:])  # First row, all columns
    ax2 = fig.add_subplot(gs[4:8:])  # First row, all columns
    ax3 = fig.add_subplot(gs[8:9,:])  # First row, all columns

    if ts == 1:
        # First, plot filled contours with contourf
        # Salinity and temperature plot
        # c_sal = np.fliplr(c_sal)
        # c_tem = np.fliplr(c_tem)
        surf_s = ax1.contourf(X, Y1, c_sal, cmap=cmo.haline, levels=contour_levels_s, vmin=0, vmax=40, zorder=2)
        # Create a mask of NaN regions, and overplot the NaN regions in gray
        # nan_mask = np.isnan(c_sal)
        mask = np.ones_like(c_sal)
        mask[np.isnan(c_sal)] = 0
        ax1.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5], alpha=1.0, zorder=1)
        # if np.any(nan_mask):
            # ax1.contourf(X, Y1, nan_mask.astype(float), colors='gray', alpha=0.5, levels=[0.5, 1.5])
        # ax1.set_title('Salinity')
        cbar_s = plt.colorbar(surf_s, ax=ax1, label='Salinity (PSU)')
        # Temperature plot  
        surf_t = ax2.contourf(X, Y1, c_tem, cmap=cmo.thermal, levels=contour_levels_t, vmin=16, vmax=32, zorder=2)
        # ax2.set_title('Temperature')
        # if np.any(nan_mask):
        #     ax2.contourf(X, Y1, nan_mask.astype(float), colors='gray', alpha=0.5, levels=[0.5, 1.5])
        ax2.contourf(X, Y1, mask, colors='gray', levels=[-0.5, 0.5, 1.5], alpha=1.0, zorder=1)
        cbar_t = plt.colorbar(surf_t, ax=ax2, label='Temperature (°C)')
        if X_length is not None:
            x_value = X[:, col_X][0] # Get the X value for the found column
            # Plot vertical red line
            ax1.axvline(x=x_value, color='red', linestyle='--', linewidth=2, label=f"Threshold @ {x_value:.2f}")
            # Convert x_value to kilometers
            x_value_km = x_value / 1000  # Convert meters to km
            # Append new data as a NumPy row
            new_entry = np.array([[ts_ds.time[ii], x_value_km, np.flip(ts_ds.lon[col_X]), np.flip(ts_ds.lat[col_X])]])  # Convert new value into a 2D NumPy array
            store_swi = np.vstack((store_swi, new_entry))  # Stack new data
            # Display the value in kilometers with two decimal places
            ax1.text(x_value + 0.2, Y1.max(), f"{x_value_km:.2f} km", fontsize=12, color='black',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            
    if hydro==1:
        # First, plot filled contours with contourf
        # surf = ax1.contourf(X, Y1, c_sal, cmap='turbo',levels=contour_levels, vmin=1, vmax=35)
        # Plot top of the water
        ax1.plot(X[0,:], hydro_ds.water_level[ii, :], 'k', linewidth=4) #hydro_ds.distance
        ax2.plot(X[0,:], hydro_ds.water_level[ii, :], 'k', linewidth=4) #hydro_ds.distance
        # Plot bathymetry
        td = hydro_ds.total_depth
        # x_values = X[0,:]
        # ax1.plot(X[0,:], hydro_ds.total_depth, 'k', linewidth=4)
        # ax2.plot(X[0,:], hydro_ds.total_depth, 'k', linewidth=4)
        # Plot quivers with a reduced number of vectors
        n = 3  # Adjust as needed for Y direction
        m = 10  # Adjust as needed for X direction
    
        # Normalize vectors
        magnitude = np.sqrt(c_uv**2 + c_vv**2)
        un = c_uv / magnitude
        vn = c_vv / magnitude
        # Replace NaN values that arise from division by zero
        un = np.nan_to_num(un)
        vn = np.nan_to_num(vn)
    
        # Determine the sign of the u component for direction
        u_direction = np.sign(c_uv[0::n, 0::m])
        v_direction = np.zeros_like(u_direction)  # No vertical component for direction
    
        # Create masks for left and right directions
        mask_left = u_direction < 0
        mask_right = u_direction > 0
    
        # Plot quivers indicating flow direction (horizontal only)
        ax1.quiver(X[0::n, 0::m][mask_right], Y1[0::n, 0::m][mask_right],
                    u_direction[mask_right], v_direction[mask_right],
                    scale=6, scale_units='inches', width=0.002, headwidth=4, headlength=2, headaxislength=1,
                    color='black', label='Out', zorder=3)
    
        ax1.quiver(X[0::n, 0::m][mask_left], Y1[0::n, 0::m][mask_left],
                    u_direction[mask_left], v_direction[mask_left],
                    scale=6, scale_units='inches', width=0.002, headwidth=4, headlength=2, headaxislength=1,
                    color='blue', label='Left', zorder=3)
        
    # Ensure there's stored data before finding max, and X,Y Position
    if store_swi.size > 0:
        max_swi_length = np.max(store_swi[:, 1])  # Get the maximum SWI length
        max_indices = np.where(store_swi[:, 1] == max_swi_length)[0]  # Get all indices where max SWI occurs
        max_index = max_indices[-1]  # Select the last occurrence of max SWI
    
        max_time, max_swi_length, max_lon, max_lat = store_swi[max_index]  # Extract values
    
        # Convert max time to readable format if necessary
        max_time_str = str(max_time)
    
        # Plot vertical magenta line for max SWI length
        ax1.axvline(x=max_swi_length * 1000, color='magenta', linestyle='--', linewidth=2,
                    label=f"Max SWI @ {max_swi_length:.2f} km")
    
        # Add a textbox with max SWI length and date/time
        ax1.text(max_swi_length * 1000 + 0.2, Y1.max() - 2,
                 f"Max Length: {max_swi_length:.2f} km\n{max_time_str}",
                 fontsize=12, color='black',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Then, add contour lines with contour
    contour_colors_ax1 = ['w', 'w', 'k', 'k', 'k', 'k', 'k', 'k', 'k']  # The colors for each contour level
    contours_s = ax1.contour(hydro_ds.X, Y1, c_sal, colors=contour_colors_ax1, levels=contour_levels_s, linewidths=2, vmin=1, vmax=40)
    contours_t = ax2.contour(hydro_ds.X, Y1, c_tem, colors=contour_colors_ax1, levels=contour_levels_t, linewidths=2, vmin=1, vmax=35)
    
    # ax1.clabel(contours_s, inline=True, fontsize=15, fmt='%1.0f', inline_spacing=2, colors=contour_colors_ax1)  # Label the contours with their values
    # ax1.clabel(contours, inline=True, fontsize=15, fmt='%1.0f', 
    #        inline_spacing=2, colors='k',
    #        bbox={'boxstyle': 'round,pad=0.3', 'edgecolor': 'none', 'facecolor': 'white'})
    
    # Adjust colorbar font size
    # cbar = plt.colorbar(surf_s, ax=ax1, label='Salinity (ppt)')
    # cbar.ax.tick_params(labelsize=20)  # Adjust tick font size
    # cbar.set_label('Salinity (ppt)', fontsize=20)  # Adjust label font size

    if measurements==1:
        # Adjust measurement data based on current water level
        for file_id, dist_point in zip(files, dist_points):
            data = measurement_data[file_id]
            c_dist = np.mean(data['Distance'])
            # Find the closest distance in the dist array
            closest_idx = np.abs(hydro_ds.dist - c_dist).argmin()
            closest_wl = hydro_ds.wl[ii, closest_idx]
    
            # Insert new depth level at the water level
            new_depth = pd.DataFrame({
                'Distance': [data['Distance'].iloc[0]],  # Copy first distance value
                'Depth': [-closest_wl],  # Use negative water level
                'Salinity': [data['Salinity'].iloc[0]],  # Copy first salinity value
                'Date': [data['Date'].iloc[0]]  # Copy first salinity value
            })
            data = pd.concat([new_depth, data], ignore_index=True)
    
            ax1.scatter(data['Distance'], data['Depth'] * -1, c=data['Salinity'],
                        cmap='turbo', vmin=1, vmax=35, edgecolors='w', s=10, zorder=5)

        # # Plot the contour lines of the interpolated salinity (from measured data)
        # all_measurement_data = pd.concat(measurement_data.values())
        # points = all_measurement_data[['Distance', 'Depth']].values
        # points[:,1] = points[:,1] * - 1
        # values = all_measurement_data['Salinity'].values
    
        # # Interpolate the salinity data onto the grid
        # salinity_grid = griddata(points, values, (X, Y1), method='cubic') # cubic, nearest
    
        # # Plot the interpolated salinity data contours
        # contour_colors = ['w', 'k', 'w', 'k', 'w', 'k', 'w', 'k', 'w']  # The colors for each contour level
        # contour_colors = ['w', 'w', 'w', 'w', 'w', 'w', 'w','w','w']  # The colors for each contour level
        # contours = ax1.contour(X, Y1, salinity_grid, levels=contour_levels_s, colors=contour_colors, linewidths=2)
        # # Label the contours with a white background
        # ax1.clabel(contours, inline=False, fontsize=15, fmt='%1.0f')  # Label the contours with their values
    
    # Adjust figure ax1
    xticks = np.arange(0, 50001, 5000)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_xticks(xticks)
    ax1.set_ylabel('Depth (m)', fontsize=18)
    ax1.tick_params(labelsize=20)
    # Format the time for the title, axis are inverted to make the mouth of river being on the right of the screen
    time_str = t[ii].strftime('%Y-%b-%d %H:%M')
    ax1.set_title(f'Time: {time_str} model used -> {sims_name} -> Branch: {river_branch}', fontsize=22)
    ax1.invert_xaxis()
    
    # Adjust figure ax1
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_xticks(xticks)
    ax2.set_xlabel('Distance along river (m)', fontsize=18, labelpad=15)
    ax2.set_ylabel('Depth (m)', fontsize=18)
    ax2.tick_params(labelsize=20)
    # Format the time for the title, axis are inverted to make the mouth of river being on the right of the screen
    time_str = t[ii].strftime('%Y-%b-%d %H:%M')
    # ax2.set_title(f'Time: {time_str} model used -> {sims_name} -> Branch: {river_branch}', fontsize=22)
    ax2.invert_xaxis()

    # # Plot river discharge data in ax3
    ax3.plot(discharge_dates, discharge_values, 'k')
    if 0 <= discharge_index < len(discharge_dates):
        ax3.plot(discharge_dates[discharge_index], discharge_values[discharge_index], 'ro')  # Highlight the current time step
    ax3.set_xlim([start_date, end_date])
    ax3.set_ylim([0,max_discharge_lim])
    ax3.set_ylabel('Discharge (m³/s)')
    ax3.grid(True)
    
    fig.tight_layout()

    figure_name = os.path.join(folder_name_plots, f'plot_{ii:04d}.png')
    print(folder_name_plots, ',', c_t)
    # plt.show()
    plt.savefig(figure_name)
    plt.close()

    image_paths.append(figure_name)
    
# save stored data from the max salinity: date, max length in km, lat and lon
f_store_swi = folder_name + f'/results_{river_branch}_stored_swi.txt'

# Open file for writing
with open(f_store_swi, 'w') as f:
    for row in store_swi:
        # Convert datetime (first column) to string format YYYY-MM-DD::hh:mm:ss
        date_str = row[0].isoformat(sep='::')  # Convert cftime datetime
        length_km = f"{row[1]:.4f}"  # Format length with 4 decimals
        lat = f"{row[2]:.8f}"  # Format latitude with 8 decimals
        lon = f"{row[3]:.8f}"  # Format longitude with 8 decimals

        # Write to file with tab delimiter
        f.write(f"{date_str}\t{length_km}\t{lat}\t{lon}\n")


# Create a video from the saved images
if make_video == 1:
    output_video = os.path.join(folder_name, f'{river_branch}_video4.mp4')
    with imageio.get_writer(output_video, mode='I', fps=4) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    print("Video saved at", output_video)
