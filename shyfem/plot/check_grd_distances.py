#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:45:31 2026

@author: utente
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

class TransectExtractor:
    def __init__(self):
        self.points = {}
        self.elements = []
        self.point_coords = {}
        self.node_adjacency = {}
        self.selected_points = None
        self.path_df = None
        
    def read_grd(self, filename):
        """Read GRD file and extract points and elements"""
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
                    points[point_id] = (x, y, point_type, z)
                
                # Parse elements (type 2)
                elif len(parts) >= 7 and parts[0] == '2':
                    element_id = int(parts[1])
                    element_type = int(parts[2])
                    num_points = int(parts[3])
                    node_ids = [int(parts[4]), int(parts[5]), int(parts[6])]
                    depth = float(parts[7]) if len(parts) > 7 else 0.0
                    elements.append((element_id, element_type, num_points, node_ids, depth))
            
            self.points = points
            self.elements = elements
            
            # Build point coordinates dictionary
            self.point_coords = {pid: (x, y) for pid, (x, y, _, _) in points.items()}
            
            # Build node adjacency
            self._build_adjacency()
            
            print(f"Loaded: {len(points)} points, {len(elements)} elements")
            return True
            
        except Exception as e:
            print(f"Error reading GRD: {e}")
            return False
    
    def _build_adjacency(self):
        """Build adjacency list from elements"""
        self.node_adjacency = {pid: set() for pid in self.point_coords.keys()}
        for element in self.elements:
            node_ids = element[3]  # Get the node_ids list
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    self.node_adjacency[node_ids[i]].add(node_ids[j])
                    self.node_adjacency[node_ids[j]].add(node_ids[i])

    def plot_distance_contour(self, distance_type='nearest', k_neighbors=10, 
                              contour_levels=20, cmap='viridis', show_points=True):
        """
        Plot contour map of distances between nodes.
        
        Parameters:
        -----------
        distance_type : str
            'nearest' - distance to nearest neighbor
            'knn' - average distance to k nearest neighbors
            'all' - average distance to all nodes
        k_neighbors : int
            Number of neighbors to average over (if distance_type='knn')
        contour_levels : int or list
            Number of contour levels or list of levels
        cmap : str
            Colormap name
        show_points : bool
            Whether to show nodes as black X markers
        """
        # Get coordinates and point IDs
        point_ids = list(self.point_coords.keys())
        coords = np.array([self.point_coords[pid] for pid in point_ids])
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Compute distances for each node
        if distance_type == 'nearest':
            # Distance to nearest neighbor
            tree = cKDTree(coords)
            distances, _ = tree.query(coords, k=2)  # k=2 gives self + nearest
            z_values = distances[:, 1]  # Skip self
            
        elif distance_type == 'knn':
            # Average distance to k nearest neighbors
            tree = cKDTree(coords)
            k = min(k_neighbors + 1, len(coords))  # Don't exceed number of points
            distances, _ = tree.query(coords, k=k)
            z_values = np.mean(distances[:, 1:], axis=1)  # Skip self
            
        elif distance_type == 'all':
            # Average distance to all other nodes
            z_values = []
            for i in range(len(coords)):
                all_dists = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
                z_values.append(np.mean(all_dists[all_dists > 1e-10]))
            z_values = np.array(z_values)
        
        # Create grid for interpolation
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Estimate grid spacing from data
        # Use 1/10 of the average of x and y ranges divided by sqrt(number of points)
        grid_resolution = ((x_max - x_min) + (y_max - y_min)) / (2 * np.sqrt(len(point_ids)))
        grid_resolution = max(grid_resolution, 1.0)  # Minimum 1 meter resolution
        
        # Create regular grid
        nx = max(10, int((x_max - x_min) / grid_resolution) + 1)
        ny = max(10, int((y_max - y_min) / grid_resolution) + 1)
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate distances onto regular grid
        zi = griddata((x, y), z_values, (xi, yi), method='linear')
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot contour map
        contour = ax.contourf(xi, yi, zi, levels=contour_levels, cmap=cmap, extend='both')
        contour_lines = ax.contour(xi, yi, zi, levels=contour_levels, colors='black', linewidths=0.5, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, label=f'Distance {distance_type} (m)', extend='both')
        cbar.ax.tick_params(labelsize=10)
        
        # Overlay points as black X
        if show_points:
            ax.scatter(x, y, marker='x', color='black', s=20, linewidth=1, alpha=0.7, zorder=5)
        
        # Add labels and title
        ax.set_xlabel('X Coordinate (m)', fontsize=12)
        ax.set_ylabel('Y Coordinate (m)', fontsize=12)
        title_map = {
            'nearest': 'Nearest Neighbor Distance',
            'knn': f'Average Distance to {k_neighbors} Nearest Neighbors',
            'all': 'Average Distance to All Nodes'
        }
        ax.set_title(f'Distance Contour Map - {title_map.get(distance_type, distance_type)}', fontsize=14)
        
        # Add grid lines
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set aspect ratio to equal
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax

    def plot_histogram_distances(self, distance_type='nearest', k_neighbors=10, bins=30):
        """
        Plot histogram of node distances.
        """
        # Get coordinates
        point_ids = list(self.point_coords.keys())
        coords = np.array([self.point_coords[pid] for pid in point_ids])
        
        # Compute distances
        if distance_type == 'nearest':
            tree = cKDTree(coords)
            distances, _ = tree.query(coords, k=2)
            z_values = distances[:, 1]
            
        elif distance_type == 'knn':
            tree = cKDTree(coords)
            k = min(k_neighbors + 1, len(coords))
            distances, _ = tree.query(coords, k=k)
            z_values = np.mean(distances[:, 1:], axis=1)
            
        elif distance_type == 'all':
            z_values = []
            for i in range(len(coords)):
                all_dists = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
                z_values.append(np.mean(all_dists[all_dists > 1e-10]))
            z_values = np.array(z_values)
        
        # Create histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(z_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Add statistics
        mean_val = np.mean(z_values)
        median_val = np.median(z_values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f} m')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.2f} m')
        
        ax.set_xlabel('Distance (m)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title_map = {
            'nearest': 'Nearest Neighbor Distance',
            'knn': f'Average Distance to {k_neighbors} Nearest Neighbors',
            'all': 'Average Distance to All Nodes'
        }
        ax.set_title(f'Distance Distribution - {title_map.get(distance_type, distance_type)}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    GRD_FILE = "/home/utente/Documenti/OMBRES/grid_ff/adri_lags_15mPiles_276714_excluded.grd"
    
    extractor = TransectExtractor()
    extractor.read_grd(GRD_FILE)
    
    # Plot nearest neighbor distance contour
    extractor.plot_distance_contour(distance_type='all', contour_levels=25)
    
    # Plot average distance to 5 nearest neighbors
    extractor.plot_distance_contour(distance_type='knn', k_neighbors=5, contour_levels=20, cmap='plasma')
    
    # Plot histogram
    extractor.plot_histogram_distances(distance_type='nearest', bins=40)