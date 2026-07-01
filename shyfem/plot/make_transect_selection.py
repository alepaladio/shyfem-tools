#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transect Extraction Tool - Standalone for Spyder
Reads GRD file, loads transect shapefile, extracts nodes along transect using connectivity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
import shapefile
import math

class TransectExtractor:
    def __init__(self, buffer_distance_meters=2350):
        """
        Initialize extractor with buffer distance in meters.
        Default buffer: 100 meters (adjust based on your grid resolution)
        """
        self.buffer_distance_meters = buffer_distance_meters
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
            self.point_coords = {pid: (x, y) for pid, (x, y, _, _) in points.items()}
            
            self._build_adjacency()
            
            print(f"Loaded: {len(points)} points, {len(elements)} elements")
            return True
            
        except Exception as e:
            print(f"Error reading GRD: {e}")
            return False
    
    def _build_adjacency(self):
        """Build node adjacency from elements"""
        adjacency = {}
        
        for _, _, _, node_ids, _ in self.elements:
            p1, p2, p3 = node_ids
            
            if p1 not in adjacency:
                adjacency[p1] = set()
            if p2 not in adjacency:
                adjacency[p2] = set()
            if p3 not in adjacency:
                adjacency[p3] = set()
            
            adjacency[p1].add(p2)
            adjacency[p1].add(p3)
            adjacency[p2].add(p1)
            adjacency[p2].add(p3)
            adjacency[p3].add(p1)
            adjacency[p3].add(p2)
        
        self.node_adjacency = {k: list(v) for k, v in adjacency.items()}
        print(f"Built adjacency: {len(self.node_adjacency)} nodes connected")
    
    def read_shapefile(self, filename):
        """Read transect line from shapefile"""
        try:
            sf = shapefile.Reader(filename)
            shapes = sf.shapes()
            
            if not shapes:
                print("No shapes found in shapefile")
                return None, 0
            
            vertices = shapes[0].points
            line_points = [(p[0], p[1]) for p in vertices]
            
            # Check for start point ID attribute
            start_id = 0
            if sf.fields:
                field_names = [f[0] for f in sf.fields[1:]]
                if 'point_id_s' in field_names:
                    idx = field_names.index('point_id_s')
                    records = sf.records()
                    if records and len(records[0]) > idx:
                        start_id = records[0][idx]
                        if start_id and start_id != 0:
                            print(f"Start point ID from attribute: {start_id}")
                            return line_points, int(start_id)
            
            print("No start point ID found in attributes, will auto-detect")
            return line_points, 0
            
        except Exception as e:
            print(f"Error reading shapefile: {e}")
            return None, 0
    
    def _calculate_distance(self, lon1, lat1, lon2, lat2):
        """Calculate distance in meters using Haversine"""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371000 * c
    
    def plot_results(self, points_df, selected_df, path_df, transect_df):
        """Plot the transect results"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.scatter(points_df['x'], points_df['y'], c='gray', s=5, alpha=0.3, label='All points')
        
        if selected_df is not None and not selected_df.empty:
            ax.scatter(selected_df['x'], selected_df['y'], c='red', s=20, alpha=0.5, label='Selected points')
        
        if path_df is not None and not path_df.empty:
            ax.scatter(path_df['x'], path_df['y'], c='blue', s=30, alpha=0.8, label='Path points')
            ax.plot(path_df['x'], path_df['y'], 'b-', linewidth=2, alpha=0.7, label='Path line')
        
        if transect_df is not None and not transect_df.empty:
            ax.plot(transect_df['x'], transect_df['y'], 'g--', linewidth=2, alpha=0.6, label='Original transect')
            ax.scatter(transect_df['x'].iloc[0], transect_df['y'].iloc[0], c='green', s=100, marker='*', label='Start')
            ax.scatter(transect_df['x'].iloc[-1], transect_df['y'].iloc[-1], c='red', s=100, marker='*', label='End')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Transect Node Extraction Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
    def create_area_of_interest(self, transect_points, buffer_distance_meters):
        """
        Create a buffer polygon around the transect line.
        Returns buffer polygon, selected point IDs with coordinates,
        and their positions along the transect.
        """
        from shapely.geometry import LineString, Point as ShapelyPoint
        from shapely.ops import nearest_points
        
        line = LineString(transect_points)
        
        # Fixed buffer distance of 250 meters
        # buffer_distance_meters = 250.0
        # Convert to degrees (approx 1 deg = 111km)
        buffer_deg = buffer_distance_meters / 111000.0
        buffer_polygon = line.buffer(buffer_deg)
        
        # Find points within buffer with their positions
        selected_points = []
        
        for pid, (x, y) in self.point_coords.items():
            point = ShapelyPoint(x, y)
            if buffer_polygon.contains(point):
                # Calculate position along transect
                pos = line.project(point)
                total_length = line.length
                
                # Normalize position (0 = start, 1 = end)
                normalized_pos = pos / total_length if total_length > 0 else 0.0
                
                # Calculate distance from point to line
                nearest = nearest_points(point, line)
                distance_to_line = self._calculate_distance(
                    x, y, nearest[1].x, nearest[1].y
                )
                
                selected_points.append({
                    'point_id': pid,
                    'x': x,
                    'y': y,
                    'position_along_transect': normalized_pos,
                    'distance_from_line_meters': distance_to_line,
                    'distance_from_start_meters': pos
                })
        
        # Sort by position along transect
        selected_points.sort(key=lambda p: p['position_along_transect'])
        # %%
        # Plot results
        # x = [p['x'] for p in selected_points]
        # y = [p['y'] for p in selected_points]
        # x_all = [coord[0] for coord in self.point_coords.values()]
        # y_all = [coord[1] for coord in self.point_coords.values()]
        
        # plt.figure(figsize=(10, 8))
        # plt.scatter(x_all, y_all, color='lightgrey', marker='.', s=15, label='All points')
        # plt.scatter(x, y, color='red', s=20, label='Selected points', zorder=5)
        
        # # Plot buffer polygon boundary
        # if buffer_polygon.geom_type == 'Polygon':
        #     x_buf, y_buf = buffer_polygon.exterior.xy
        #     plt.plot(x_buf, y_buf, 'g--', linewidth=1, alpha=0.5, label=f'Buffer ({buffer_distance_meters}m)')
        
        # # Plot transect line
        # x_tra = [p[0] for p in transect_points]
        # y_tra = [p[1] for p in transect_points]
        # plt.plot(x_tra, y_tra, 'b-', linewidth=2, label='Transect')
        
        # plt.axis('equal')
        # plt.xlim(12.90,13.10)
        # plt.ylim(43.85,44)
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # plt.title(f'Selected {len(selected_points)} points within {buffer_distance_meters}m buffer zone')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        # %%
        print(f"Selected {len(selected_points)} points within {buffer_distance_meters}m buffer zone")
        return buffer_polygon, selected_points
    
    def compute_azimuth(self, cx, cy, sel_x, sel_y):
        """
        Computes aimuth from 2 points
        """
        dx = sel_x - cx
        dy = sel_y - cy
        current_azimuth = math.degrees(math.atan2(dx, dy))
        current_azimuth = (current_azimuth + 360) % 360
        return current_azimuth
    
    def check_azimuth_path(self, original_points, combined_points, tolerance=30):
        """
        Check if combined points follow the same direction as original path.
        If deviations > tolerance, reorder by distance along original path.
        
        Args:
            original_points: List of (x,y) from original transect
            combined_points: List of (x,y) to check
            tolerance: Degrees tolerance (default 30°)
        
        Returns:
            Fixed list of points in correct order
        """
        # Calculate original direction (azimuth from first to last)
        dx = original_points[-1][0] - original_points[0][0]
        dy = original_points[-1][1] - original_points[0][1]
        original_azimuth = math.degrees(math.atan2(dx, dy)) % 360
        
        # Calculate azimuth for each combined point
        combined_array = np.array(combined_points)
        dx = np.diff(combined_array[:, 0])
        dy = np.diff(combined_array[:, 1])
        azimuths = np.degrees(np.arctan2(dx, dy))
        azimuths = (azimuths + 360) % 360
        
        # Check if any point deviates more than tolerance
        for i, az in enumerate(azimuths):
            diff = abs(az - original_azimuth)
            diff = min(diff, 360 - diff)
            if diff > tolerance:
                print(f"Point {i} deviates {diff:.1f}° from original path")
                print(f"Reordering by distance along original path...")
                
                # Reorder by distance along original path
                line = LineString(original_points)
                ordered = []
                for point in combined_points:
                    p = Point(point)
                    pos = line.project(p)
                    ordered.append((pos, point))
                
                ordered.sort(key=lambda x: x[0])
                return [p[1] for p in ordered]
        
        print("Path order is consistent with original")
        return combined_points
    
    def densify_line(self, points, num_segments=1000):
        """
        Densify a line by dividing it into equal segments, maintaining original order.
        
        Args:
            points: List of (x, y) coordinates defining the line
            num_segments: Number of segments to divide the line into
        
        Returns:
            List of (x, y) coordinates for the densified line (starts at point[0], ends at point[-1])
        """
        if len(points) < 2:
            print('A line needs more than 1 point.')
            return points
        
        # Convert to numpy array
        points_array = np.array(points)
        
        plot_check=0
        
        if plot_check==1:
            # Plot lines to check that they are going in 1 way
            plt.figure(figsize=(10, 8))
            plt.axis('equal')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Transect Progression')
            
            # Plot path and current point
            for i in range(len(points_array)):
                # Plot all points up to current (the path so far)
                plt.plot(points_array[:i+1, 0], points_array[:i+1, 1], 'b-', alpha=0.5)
                # Plot current point in red
                plt.plot(points_array[i, 0], points_array[i, 1], 'ro', markersize=10)
                plt.pause(0.005)
                print(i)
        
        # Calculate cumulative distances along the line
        diffs = np.diff(points_array, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_distances = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cumulative_distances[-1]
        
        # Generate equally spaced points along the line
        target_distances = np.linspace(0, total_length, num_segments + 1)
        
        # Interpolate points at target distances
        densified_points = []
        for target_distance in target_distances:
            # Find which segment this point falls in
            idx = np.searchsorted(cumulative_distances, target_distance)
            if idx == 0:
                densified_points.append(tuple(points_array[0]))
            elif idx >= len(points):
                densified_points.append(tuple(points_array[-1]))
            else:
                # Interpolate between points idx-1 and idx
                segment_length = cumulative_distances[idx] - cumulative_distances[idx-1]
                if segment_length > 0:
                    t = (target_distance - cumulative_distances[idx-1]) / segment_length
                    interpolated = points_array[idx-1] + t * (points_array[idx] - points_array[idx-1])
                    densified_points.append(tuple(interpolated))
                else:
                    densified_points.append(tuple(points_array[idx-1]))
                    
        combined_points = sorted(set(points + densified_points), key=lambda p: p[0])  # If sorted by x
        combined_points_check = self.check_azimuth_path(points_array, combined_points)
        
        # Checking plots
        if plot_check==1:
            # Plot check
            combined_points = np.array(combined_points_check)
            if plot_check==1:
                # Plot lines to check that they are going in 1 way
                plt.figure(figsize=(10, 8))
                plt.axis('equal')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title('Transect Progression')
                
                # Plot path and current point
                for i in range(len(combined_points)):
                    # Plot all points up to current (the path so far)
                    plt.plot(combined_points[:i+1, 0], combined_points[:i+1, 1], 'b-', alpha=0.5)
                    # Plot current point in red
                    plt.plot(combined_points[i, 0], combined_points[i, 1], 'ro', markersize=10)
                    plt.pause(0.01)
                    print(i)
        # %%
        print('finish checking')
        return combined_points_check

    def get_radius_circle(self, cx, cy, max_distance):
        lat_rad = math.radians(cy)
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(lat_rad)
        # Number of points around circle
        theta = np.linspace(0, 2*np.pi, 100)
        # compute other cicle things
        radius_deg_lat = max_distance / meters_per_deg_lat
        radius_deg_lon = max_distance / meters_per_deg_lon
        circle_lon = cx + radius_deg_lon * np.cos(theta)
        circle_lat = cy + radius_deg_lat * np.sin(theta)
        return circle_lon, circle_lat
    
    def build_path_from_selected_points(self, start_node, end_node, selected_ids, transect_with_azimuth):
        """
        Build path using only points within the buffer zone.
        Uses connectivity with angle-based selection.
        """
        plot_check=0
        
        # Quick return if start equals end
        if start_node == end_node:
            return [start_node]
        
        # Convert to dict for fast lookup by point_id
        points_dict = {p['point_id']: p for p in selected_ids}
        
        # Validate start and end nodes
        if start_node not in points_dict:
            print(f"ERROR: Start node {start_node} not found in selected points")
            return []
        if end_node not in points_dict:
            print(f"ERROR: End node {end_node} not found in selected points")
            return []
        
        # Get start and end positions
        start_pos = points_dict[start_node]
        end_pos = points_dict[end_node]
        
        # Build path
        path = [start_node]
        visited = {start_node}
        current_node = start_node
        current_pos = start_pos
        
        # Transect array for quick access
        transect_array = np.array(transect_with_azimuth)
        transect_xy = transect_array[:, :2]  # x, y columns
        
        # Maximum iterations = number of selected points (safety limit)
        max_iterations = len(selected_ids) * 2
        iterations = 0
        
        # Start loop, 
        # 1. get current coordinates (cx,cy), 
        # 2. from candidates compute distance and azimuth from current coordinates
        # 2.1 get candidate position, avoid previous visited, check if end node is here to close path
        # 3. Compute max_distance from (cx,cy) to all available candidates
        # 4. For (cx,cy) to max_distance circle, get possible transect_xy
        # 4.1 Always from inside this circle, get the last node, and compute azimuth and distance
        # 5. From selected piece of transect, always use last point inside circle
        # 6. Compute azimuth from selected transect INSIDE the current circle
        # 7. select the best node based on all candidates azimuth vs current azimuth (from current transect).
        # 8. Add best candidate to path. 
        while iterations < max_iterations and current_node != end_node:
            iterations += 1
            if iterations==50:
                print(iterations)
            # Get current coordinates
            cx = current_pos['x']
            cy = current_pos['y']
            
            # Step 1: Get available connected nodes (unvisited, in selected_ids)
            connected_nodes = self.node_adjacency.get(current_node, [])
            available = [n for n in connected_nodes if n in points_dict and n not in visited]
            
            if not available:
                print(f"WARNING: No available nodes from {current_node}, backtracking...")
                if len(path) > 1:
                    path.pop()
                    current_node = path[-1]
                    current_pos = points_dict[current_node]
                    continue
                break
            
            # Step 2: Build candidate list with distances and azimuths
            candidates = []
            for node_id in available:
                node_pos = points_dict[node_id]
                
                # Skip if this is the end node - we want to reach it!
                if node_id == end_node:
                    path.append(node_id)
                    print(f"Reached end node {end_node}!")
                    return path
                
                # Calculate distance
                dist = self._calculate_distance(
                    current_pos['x'], current_pos['y'],
                    node_pos['x'], node_pos['y']
                )
                
                # Calculate azimuth from current to candidate
                dx = node_pos['x'] - current_pos['x']
                dy = node_pos['y'] - current_pos['y']
                azimuth_deg = math.degrees(math.atan2(dx, dy))
                azimuth_deg = (azimuth_deg + 360) % 360
                
                candidates.append({
                    'point_id': node_id,
                    'x': node_pos['x'],
                    'y': node_pos['y'],
                    'distance': dist,
                    'azimuth_deg': azimuth_deg
                })
            
            # If no candidates, break
            if not candidates:
                break
            
            # Step 3: Find max distance from candidates
            max_distance = max([c['distance'] for c in candidates])
            
            # Step 4: Find transect points within max_distance circle
            circle_points = []
            for i in range(len(transect_array)):
                tx, ty, _ = transect_array[i]
                dist_to_current = self._calculate_distance(cx, cy, tx, ty)
                if dist_to_current <= max_distance:
                    circle_points.append({'index': i, 'distance': dist_to_current})
            
            if not circle_points:
                print("WARNING: No transect points within circle, using closest candidate")
                best_candidate = min(candidates, key=lambda c: c['distance'])
                path.append(best_candidate['point_id'])
                visited.add(best_candidate['point_id'])
                current_node = best_candidate['point_id']
                current_pos = points_dict[current_node]
                continue
            # %%
            # Plot check see available points from transect, in circle from nodes
            if plot_check==1:
                # Plot available nodes
                x_selt = [p['x'] for p in selected_ids]
                y_selt = [p['y'] for p in selected_ids]
                plt.scatter(x_selt,y_selt,color='lightgray',marker='x',s=20)
                # Plot current node 
                plt.plot(cx, cy, 'or')
                # Plot corresponding circle
                circle_lon, circle_lat = self.get_radius_circle(cx, cy, max_distance)
                plt.plot(circle_lon, circle_lat, 'g--', linewidth=2, alpha=0.8, label='Search circle', zorder=3)
                # Plot transect nodes that fall in circle
                x_c_trans = [transect_array[p['index']][0] for p in circle_points]
                y_c_trans = [transect_array[p['index']][1] for p in circle_points]
                plt.plot(x_c_trans,y_c_trans, 'xr')
                
                plt.axis('equal')
                plt.xlim(12.90,13.10)
                plt.ylim(43.80,44.00)
            # %%
            # Step 5: Select transect index (inside or after circle)
            max_idx = max([p['index'] for p in circle_points])
            # Select always last point inside circle, which in this case is the fartest point.
            selected_idx = max_idx  
            
            # Step 6: Get selected transect point (transect_array is always from start to end)
            # and compute current azimuth
            selected_tx, selected_ty, _ = transect_array[selected_idx]
            
            dx = selected_tx - cx
            dy = selected_ty - cy
            current_azimuth = math.degrees(math.atan2(dx, dy))
            current_azimuth = (current_azimuth + 360) % 360
            
            # Step 7: Select candidate with azimuth closest to current_azimuth
            best_candidate = None
            best_angle_diff = float('inf')
            # nodes azimuth is in candidates, current transect azimuth is in current_azimuth
            for candidate in candidates:
                angle_diff = abs(candidate['azimuth_deg'] - current_azimuth)
                angle_diff = min(angle_diff, 360 - angle_diff)
                
                if angle_diff < best_angle_diff:
                    best_angle_diff = angle_diff
                    best_candidate = candidate
            
            if best_candidate is None:
                print(f"WARNING: No candidate for node {current_node}")
                if len(path) > 1:
                    path.pop()
                    current_node = path[-1]
                    current_pos = points_dict[current_node]
                    continue
                break
            
            # Step 8: Add best candidate to path
            path.append(best_candidate['point_id'])
            visited.add(best_candidate['point_id'])
            current_node = best_candidate['point_id']
            current_pos = points_dict[current_node]
            # %%
            # Visually check circles
            if plot_check==1:
                # Plot available nodes
                x_selt = [p['x'] for p in selected_ids]
                y_selt = [p['y'] for p in selected_ids]
                plt.scatter(x_selt,y_selt,color='lightgray',marker='x',s=20)
                # Plot current node 
                plt.plot(cx, cy, 'or')
                # Plot Transect up to now
                points_lookup = {p['point_id']: (p['x'], p['y']) for p in selected_ids}
                # x_c_path = {p['point_id']: (p['x']) for p in selected_ids}
                x_c_path = [points_lookup[pid][0] for pid in path]
                y_c_path = [points_lookup[pid][1] for pid in path]
                plt.plot(x_c_path, y_c_path,'--b')
                # Plot transect
                plt.plot(transect_xy[:,0],transect_xy[:,1],'k')
                plt.scatter(transect_xy[:,0],transect_xy[:,1], marker='x', s=10)
                
                circle_lon, circle_lat = self.get_radius_circle(cx, cy, max_distance)
                plt.plot(circle_lon, circle_lat, 'g--', linewidth=2, alpha=0.8, label='Search circle', zorder=3)
                
                plt.axis('equal')
                plt.xlim(12.90,13.10)
                plt.ylim(43.80,44.00)
            # %%
            # Optional: plot progress
            if iterations % 50 == 0:
                print(f"Step {iterations}: Node {current_node}, "
                      f"Dist to end: {self._calculate_distance(current_pos['x'], current_pos['y'], end_pos['x'], end_pos['y']):.1f}m")
        
        print(f"Path built with {len(path)} nodes after {iterations} iterations")
        return path
    
    def extract_transect(self, grd_file, shapefile, start_point_id=0):
        """Main function to extract transect"""
        if not self.read_grd(grd_file):
            print("Failed to read GRD file")
            return None
           
        # Extract all GRD points
        x_all = [coord[0] for coord in self.point_coords.values()]
        y_all = [coord[1] for coord in self.point_coords.values()]

        transect_points, start_id = self.read_shapefile(shapefile)
        if transect_points is None or len(transect_points) < 2:
            print("Failed to read shapefile")
            return None
        
        # Densify the line, transect_line passes to be treated as the main line, since 
        # it adds the current used defined and a 1000 more points.
        transect_line = self.densify_line(transect_points, num_segments=1500)
        transect_array = np.array(transect_line)
        # %%
        # Compute azimuth for each point (bearing from previous to current)
        dx = np.diff(transect_array[:, 0])  # Longitude differences
        dy = np.diff(transect_array[:, 1])  # Latitude differences
        
        # Compute azimuth in degrees (0 = North, clockwise)
        azimuth_rad = np.arctan2(dx, dy)
        azimuth_deg = np.degrees(azimuth_rad)
        azimuth_deg = (azimuth_deg + 360) % 360
        
        # Create transect_azimuth array (same length as transect_array)
        transect_azimuth = np.full(len(transect_array), np.nan)
        transect_azimuth[:-1] = azimuth_deg  # All points except last
        transect_azimuth[-1] = np.nan  # Last point has no forward direction
        # %%
        # Add as column to transect_array if needed
        transect_with_azimuth = np.column_stack((transect_array, transect_azimuth))
        if start_point_id != 0:
            start_id = start_point_id
        
        # Create buffer zone
        buffer_polygon, selected_points = self.create_area_of_interest(transect_points, self.buffer_distance_meters)
        
        # Find start node
        if start_id != 0 and start_id in self.point_coords:
            start_node = start_id
        else:
            start_pos = transect_points[0]
            min_dist = float('inf')
            start_node = None
            for pid, (x, y) in self.point_coords.items():
                dist = self._calculate_distance(x, y, start_pos[0], start_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    start_node = pid
        
        # Find end node
        end_pos = transect_points[-1]
        min_dist = float('inf')
        end_node = None
        for pid, (x, y) in self.point_coords.items():
            dist = self._calculate_distance(x, y, end_pos[0], end_pos[1])
            if dist < min_dist:
                min_dist = dist
                end_node = pid
        
        print(f"Start node: {start_node}, End node: {end_node}")
        
        # Find the point with matching point_id
        start_point = next((p for p in selected_points if p['point_id'] == start_node), None)
        ending_point = next((p for p in selected_points if p['point_id'] == end_node), None)
        if start_point:
            x_start = start_point['x']
            y_start = start_point['y']
            x_end = ending_point['x']
            y_end = ending_point['y']
        else:
            print(f"Node {start_node} not found in selected_points")
        # %%
        # # Quick way to plot selected points
        # x_sel = [p['x'] for p in selected_points]
        # y_sel = [p['y'] for p in selected_points]
        # x_tra = [p[0] for p in transect_points] 
        # y_tra = [p[1] for p in transect_points]
        # x_tra_all = transect_array[:, 0]
        # y_tra_all = transect_array[:, 1]
        # # plot the points that will be available for selection
        # plt.plot(x_sel, y_sel, 'xk', markersize=5)
        # # plot the original transect line
        # plt.plot(x_tra, y_tra, 'b')
        # # plot new line
        # plt.scatter(x_tra_all,y_tra_all,color='k',marker='x', s=10)
        # # plot all points of the grid
        # plt.scatter(x_all,y_all,color='lightgray',marker='x', s=10)
        # # plo start and end point
        # plt.scatter(x_start,y_start,color='r',marker='s',s=20)
        # plt.scatter(x_end,y_end,color='r',marker='s',s=20)
        # plt.axis('equal')
        # plt.xlim(12.75,13.50)
        # plt.ylim(43.75,44.50)
        # %%
        
        path_nodes = self.build_path_from_selected_points(start_node, end_node, selected_points, transect_with_azimuth)
        
        if not path_nodes or len(path_nodes) < 2:
            print("Failed to build path")
            return None
        
        # Create DataFrames for visualization
        path_coords = [self.point_coords[pid] for pid in path_nodes]
        self.path_df = pd.DataFrame(path_coords, columns=['x', 'y'])
        self.path_df['point_id'] = path_nodes
        self.path_df['sequence'] = range(1, len(path_nodes) + 1)
        
        selected_coords = [{'x': p['x'], 'y': p['y']} for p in selected_points]
        selected_df = pd.DataFrame(selected_coords, columns=['x', 'y'])
        selected_df['point_id'] = selected_points
        
        all_points = [(pid, x, y) for pid, (x, y) in self.point_coords.items()]
        points_df = pd.DataFrame(all_points, columns=['point_id', 'x', 'y'])
        transect_df = pd.DataFrame(transect_points, columns=['x', 'y'])
        
        self.plot_results(points_df, selected_df, self.path_df, transect_df)
        
        return path_nodes
    

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    GRD_FILE = "/home/utente/Documenti/OMBRES/grid_ff/adri_lags_15mPiles_276714_excluded.grd"
    SHAPEFILE = "/home/utente/Documenti/OMBRES/QGIS/line_along_adriatic.shp"
    # SHAPEFILE = "/home/utente/Documenti/OMBRES/QGIS/line3_north_adriatic.shp"
    
    # Set buffer distance in meters (adjust based on your needs)
    extractor = TransectExtractor(buffer_distance_meters=1550)
    path = extractor.extract_transect(GRD_FILE, SHAPEFILE, start_point_id=0)
    
    if path:
        print(f"\n✅ Transect extraction complete! Found {len(path)} nodes.")
        print("\nFirst 10 nodes:")
        for i, pid in enumerate(path[:10]):
            print(f"  {i+1}. Point ID: {pid}")