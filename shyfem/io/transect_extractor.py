"""
Transect Extraction Tool for SHYFEM grids
Extracts nodes along a user-defined transect line from a GRD file
"""

import os
import numpy as np
import pandas as pd
import math
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import shapefile


class TransectExtractor:
    """
    Extract nodes from GRD file along a transect line defined by shapefile.
    Uses mesh connectivity to build a continuous path.
    """
    
    def __init__(self, buffer_distance_meters=1550):
        """
        Initialize extractor with buffer distance in meters.
        
        Parameters
        ----------
        buffer_distance_meters : float
            Buffer distance for selecting points near transect (default: 1550m)
        """
        self.buffer_distance_meters = buffer_distance_meters
        self.points = {}
        self.elements = []
        self.point_coords = {}
        self.point_depths = {}
        self.node_adjacency = {}
        self.path_df = None
        
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
            
            self.points = points
            self.elements = elements
            
            # Check if points have depth (z) values
            points_have_depth = any(p['depth'] != 0.0 for p in points.values())
            
            # If points don't have depth, interpolate from elements
            if not points_have_depth and elements:
                print("Points have no depth, interpolating from elements...")
                self._interpolate_depth_from_elements()
            else:
                print(f"Points already have depth data")
            
            self.point_coords = {pid: (p['x'], p['y']) for pid, p in points.items()}
            self.point_depths = {pid: p['depth'] for pid, p in points.items()}
            
            self._build_adjacency()
            
            print(f"Loaded: {len(points)} points, {len(elements)} elements")
            return True
            
        except Exception as e:
            print(f"Error reading GRD: {e}")
            return False
    
    def _interpolate_depth_from_elements(self):
        """
        Interpolate depth from elements to points.
        For each point, find the average depth of all elements that contain it.
        """
        # Initialize depth accumulator for each point
        depth_sum = {pid: 0.0 for pid in self.points}
        depth_count = {pid: 0 for pid in self.points}
        
        # For each element, add its depth to each of its nodes
        for element in self.elements:
            element_depth = element['depth']
            for node_id in element['node_ids']:
                if node_id in depth_sum:
                    depth_sum[node_id] += element_depth
                    depth_count[node_id] += 1
        
        # Calculate average depth for each point
        for pid in self.points:
            if depth_count[pid] > 0:
                self.points[pid]['depth'] = depth_sum[pid] / depth_count[pid]
            else:
                self.points[pid]['depth'] = 0.0
        
        print(f"Interpolated depths for {len([p for p in self.points if self.points[p]['depth'] > 0])} points")
    
    def _build_adjacency(self):
        """Build node adjacency from elements."""
        adjacency = {}
        
        for element in self.elements:
            node_ids = element['node_ids']
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
        """Read transect line from shapefile."""
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
        """Calculate distance in meters using Haversine."""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371000 * c
    
    def densify_line(self, points, num_segments=1500):
        """
        Densify a line by dividing it into equal segments.
        
        Parameters
        ----------
        points : list of (x,y)
            Original line points
        num_segments : int
            Number of segments to divide into
        
        Returns
        -------
        list
            Densified points including original points
        """
        if len(points) < 2:
            return points
        
        points_array = np.array(points)
        
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
            idx = np.searchsorted(cumulative_distances, target_distance)
            if idx == 0:
                densified_points.append(tuple(points_array[0]))
            elif idx >= len(points):
                densified_points.append(tuple(points_array[-1]))
            else:
                segment_length = cumulative_distances[idx] - cumulative_distances[idx-1]
                if segment_length > 0:
                    t = (target_distance - cumulative_distances[idx-1]) / segment_length
                    interpolated = points_array[idx-1] + t * (points_array[idx] - points_array[idx-1])
                    densified_points.append(tuple(interpolated))
                else:
                    densified_points.append(tuple(points_array[idx-1]))
        
        # Combine and remove duplicates
        combined = points + densified_points
        combined_points = list(dict.fromkeys(combined))  # Remove duplicates preserving order
        
        return combined_points
    
    def create_area_of_interest(self, transect_points):
        """
        Create a buffer polygon around the transect line.
        
        Returns
        -------
        tuple
            (buffer_polygon, selected_points_list)
        """
        from shapely.geometry import LineString, Point as ShapelyPoint
        from shapely.ops import nearest_points
        
        line = LineString(transect_points)
        
        # Convert buffer distance to degrees (approx 1 deg = 111km)
        buffer_deg = self.buffer_distance_meters / 111000.0
        buffer_polygon = line.buffer(buffer_deg)
        
        # Find points within buffer with their positions
        selected_points = []
        
        for pid, (x, y) in self.point_coords.items():
            point = ShapelyPoint(x, y)
            if buffer_polygon.contains(point):
                pos = line.project(point)
                total_length = line.length
                normalized_pos = pos / total_length if total_length > 0 else 0.0
                
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
        
        # print(f"Selected {len(selected_points)} points within {self.buffer_distance_meters}m buffer zone")
        return buffer_polygon, selected_points
    
    def build_path_from_selected_points(self, start_node, end_node, selected_ids, transect_with_azimuth):
        """Build path using only points within the buffer zone."""
        
        # Quick return if start equals end
        if start_node == end_node:
            return [start_node]
        
        # Convert to dict for fast lookup
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
        transect_xy = transect_array[:, :2]
        
        # Maximum iterations safety limit
        max_iterations = len(selected_ids) * 2
        iterations = 0
        
        while iterations < max_iterations and current_node != end_node:
            iterations += 1
            
            # Get current coordinates
            cx = current_pos['x']
            cy = current_pos['y']
            
            # Get available connected nodes (unvisited, in selected_ids)
            connected_nodes = self.node_adjacency.get(current_node, [])
            available = [n for n in connected_nodes if n in points_dict and n not in visited]
            
            if not available:
                print(f"No available nodes from {current_node}, backtracking...")
                if len(path) > 1:
                    path.pop()
                    current_node = path[-1]
                    current_pos = points_dict[current_node]
                    continue
                break
            
            # Build candidate list with distances and azimuths
            candidates = []
            for node_id in available:
                node_pos = points_dict[node_id]
                
                if node_id == end_node:
                    path.append(node_id)
                    print(f"Reached end node {end_node}!")
                    return path
                
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
            
            if not candidates:
                break
            
            # Find max distance from candidates
            max_distance = max([c['distance'] for c in candidates])
            
            # Find transect points within max_distance circle
            circle_points = []
            for i in range(len(transect_array)):
                tx, ty, _ = transect_array[i]
                dist_to_current = self._calculate_distance(cx, cy, tx, ty)
                if dist_to_current <= max_distance:
                    circle_points.append({'index': i, 'distance': dist_to_current})
            
            if not circle_points:
                best_candidate = min(candidates, key=lambda c: c['distance'])
                path.append(best_candidate['point_id'])
                visited.add(best_candidate['point_id'])
                current_node = best_candidate['point_id']
                current_pos = points_dict[current_node]
                continue
            
            # Select transect index (always last point inside circle)
            max_idx = max([p['index'] for p in circle_points])
            selected_idx = max_idx
            
            # Get selected transect point and compute current azimuth
            selected_tx, selected_ty, _ = transect_array[selected_idx]
            
            dx = selected_tx - cx
            dy = selected_ty - cy
            current_azimuth = math.degrees(math.atan2(dx, dy))
            current_azimuth = (current_azimuth + 360) % 360
            
            # Select candidate with azimuth closest to current_azimuth
            best_candidate = None
            best_angle_diff = float('inf')
            
            for candidate in candidates:
                angle_diff = abs(candidate['azimuth_deg'] - current_azimuth)
                angle_diff = min(angle_diff, 360 - angle_diff)
                
                if angle_diff < best_angle_diff:
                    best_angle_diff = angle_diff
                    best_candidate = candidate
            
            if best_candidate is None:
                if len(path) > 1:
                    path.pop()
                    current_node = path[-1]
                    current_pos = points_dict[current_node]
                    continue
                break
            
            # Add best candidate to path
            path.append(best_candidate['point_id'])
            visited.add(best_candidate['point_id'])
            current_node = best_candidate['point_id']
            current_pos = points_dict[current_node]
        
        print(f"Path built with {len(path)} nodes after {iterations} iterations")
        return path
    
    def extract_transect(self, grd_file, shapefile, start_point_id=0):
        """
        Main function to extract transect from GRD file.
        
        Parameters
        ----------
        grd_file : str
            Path to GRD file
        shapefile : str
            Path to shapefile with transect line
        start_point_id : int
            Optional start point ID (auto-detect if 0)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: node, longitude, latitude
        """
        if not self.read_grd(grd_file):
            print("Failed to read GRD file")
            return None
        
        # Read transect from shapefile
        transect_points, start_id = self.read_shapefile(shapefile)
        if transect_points is None or len(transect_points) < 2:
            print("Failed to read shapefile")
            return None
        
        # Densify the line
        transect_line = self.densify_line(transect_points, num_segments=1500)
        transect_array = np.array(transect_line)
        
        # Compute azimuth for each point
        dx = np.diff(transect_array[:, 0])
        dy = np.diff(transect_array[:, 1])
        azimuth_rad = np.arctan2(dx, dy)
        azimuth_deg = np.degrees(azimuth_rad)
        azimuth_deg = (azimuth_deg + 360) % 360
        
        transect_azimuth = np.full(len(transect_array), np.nan)
        transect_azimuth[:-1] = azimuth_deg
        transect_azimuth[-1] = np.nan
        
        transect_with_azimuth = np.column_stack((transect_array, transect_azimuth))
        
        # Use provided start ID or auto-detect
        if start_point_id != 0:
            start_id = start_point_id
        
        # Create buffer zone and select points
        buffer_polygon, selected_points = self.create_area_of_interest(transect_points)
        
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
        
        # Build path
        path_nodes = self.build_path_from_selected_points(
            start_node, end_node, selected_points, transect_with_azimuth
        )
        
        if not path_nodes or len(path_nodes) < 2:
            print("Failed to build path")
            return None
        
        # Create DataFrame with node, longitude, latitude, depth
        path_coords = []
        for pid in path_nodes:
            if pid in self.point_coords:
                x, y = self.point_coords[pid]
                depth = self.point_depths.get(pid, 0.0)  # Get depth or default to 0
                path_coords.append({
                    'node': pid, 
                    'longitude': x, 
                    'latitude': y, 
                    'depth': depth
                })
        
        self.path_df = pd.DataFrame(path_coords)
        
        print(f"Transect extraction complete! Found {len(self.path_df)} nodes.")
        return self.path_df
    
    def save_transect_file(self, output_file):
        """Save transect data to DAT file (CSV format) with depth."""
        if self.path_df is None or self.path_df.empty:
            print("No transect data to save")
            return False
        
        # Save with header: node,longitude,latitude,depth
        self.path_df.to_csv(output_file, index=False, sep=',')
        print(f"Saved transect to: {output_file}")
        return True


def extract_transect_from_files(grd_file, shapefile, output_file, buffer_distance=1550, start_point_id=0):
    """
    Convenience function to extract transect and save to file.
    
    Parameters
    ----------
    grd_file : str
        Path to GRD file
    shapefile : str
        Path to shapefile with transect line
    output_file : str
        Path to output DAT file
    buffer_distance : float
        Buffer distance in meters (default: 1550)
    start_point_id : int
        Optional start point ID
    
    Returns
    -------
    pd.DataFrame
        Transect data with columns: node, longitude, latitude
    """
    extractor = TransectExtractor(buffer_distance_meters=buffer_distance)
    transect_df = extractor.extract_transect(grd_file, shapefile, start_point_id)
    
    if transect_df is not None:
        extractor.save_transect_file(output_file)
    
    return transect_df
