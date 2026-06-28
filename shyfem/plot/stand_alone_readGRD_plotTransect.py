#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transect Extraction Tool - Standalone for Spyder
Reads GRD file, loads transect shapefile, extracts nodes along transect using connectivity
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import shapefile  # pyshp library

class TransectExtractor:
    def __init__(self):
        self.points = {}        # {point_id: (x, y, point_type, z)}
        self.elements = []      # [(element_id, element_type, num_points, node_ids, depth)]
        self.node_adjacency = {}  # {node_id: [connected_node_ids]}
        self.point_coords = {}   # {point_id: (x, y)}
        
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
                return None
            
            vertices = shapes[0].points
            line_points = [(p[0], p[1]) for p in vertices]
            
            if sf.fields:
                field_names = [f[0] for f in sf.fields[1:]]
                if 'point_id_s' in field_names:
                    idx = field_names.index('point_id_s')
                    records = sf.records()
                    if records and len(records[0]) > idx:
                        start_id = records[0][idx]
                        if start_id and start_id != 0:
                            print(f"Start point ID from attribute: {start_id}")
                            return line_points, start_id
            
            print("No start point ID found in attributes, will auto-detect")
            return line_points, 0
            
        except Exception as e:
            print(f"Error reading shapefile: {e}")
            return None
    
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
    
    def _plot_search_step(self, current_node, unvisited, max_distance, azimuth_line, 
                       best_node, fallback_node, transect_line, current_transect, step_num, line2line):
        """Plot each search step"""
        cx, cy = self.point_coords[current_node]
        
        lat_rad = math.radians(cy)
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(lat_rad)
        
        radius_deg_lat = max_distance / meters_per_deg_lat
        radius_deg_lon = max_distance / meters_per_deg_lon
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all mesh nodes
        if self.point_coords:
            xs = [x for x, y in self.point_coords.values()]
            ys = [y for x, y in self.point_coords.values()]
            ax.scatter(xs, ys, c='k', marker='x', s=20, label='Mesh nodes', zorder=1)
        
        # Plot transect line
        if transect_line:
            lx = [p[0] for p in transect_line]
            ly = [p[1] for p in transect_line]
            ax.plot(lx, ly, 'b-', linewidth=2, alpha=0.6, label='Transect line', zorder=2)
        
        # Plot current transect path
        if len(current_transect) > 1:
            lcx = [p[1] for p in current_transect]
            lcy = [p[2] for p in current_transect]
            ax.plot(lcx, lcy, 'r-', linewidth=2, alpha=0.6, label='Current path', zorder=2)
        
        # Plot start and end of transect
        if transect_line:
            start = transect_line[0]
            end = transect_line[-1]
            ax.scatter(start[0], start[1], c='blue', s=100, marker='s', label='Line start', zorder=3)
            ax.scatter(end[0], end[1], c='blue', s=100, marker='*', label='Line end', zorder=3)
        
        # Plot current node
        ax.scatter(cx, cy, c='green', s=150, marker='o', edgecolor='black', linewidth=2, label='Current node', zorder=5)
        
        # Plot search circle
        if max_distance > 0:
            theta = np.linspace(0, 2*np.pi, 100)
            circle_lon = cx + radius_deg_lon * np.cos(theta)
            circle_lat = cy + radius_deg_lat * np.sin(theta)
            ax.plot(circle_lon, circle_lat, 'g--', linewidth=2, alpha=0.8, label='Search circle', zorder=3)
        
        # Plot candidate nodes
        candidate_nodes = []
        for node_id in unvisited:
            if node_id in self.point_coords:
                nx, ny = self.point_coords[node_id]
                dist = self._calculate_distance(cx, cy, nx, ny)
                if dist <= max_distance:
                    candidate_nodes.append((node_id, nx, ny))
        
        if candidate_nodes:
            cx_vals = [n[1] for n in candidate_nodes]
            cy_vals = [n[2] for n in candidate_nodes]
            ax.scatter(cx_vals, cy_vals, c='red', s=60, marker='o', edgecolor='black', linewidth=1, label='Candidate nodes', zorder=4)
            for node_id, nx, ny in candidate_nodes:
                ax.annotate(str(node_id), (nx, ny), fontsize=7, xytext=(3, 3), textcoords='offset points', alpha=0.7)
        
        # Highlight best node
        if best_node is not None and best_node in self.point_coords:
            bx, by = self.point_coords[best_node]
            ax.scatter(bx, by, c='green', s=200, marker='*', edgecolor='black', linewidth=2, label='Best node', zorder=6)
            ax.annotate(f'BEST: {best_node}', (bx, by), fontsize=10, fontweight='bold', xytext=(10, 10), textcoords='offset points', color='darkgreen')
        
        # Draw azimuth line
        if max_distance > 0:
            end_lon = cx + radius_deg_lon * 2 * math.cos(azimuth_line)
            end_lat = cy + radius_deg_lat * 2 * math.sin(azimuth_line)
            ax.plot([cx, end_lon], [cy, end_lat], 'g-', linewidth=1, alpha=0.5, zorder=2, label='Azimuth direction')
        
        # Plot line2line if provided
        if line2line is not None and len(line2line) == 2:
            l2l_x = [line2line[0][0], line2line[1][0]]
            l2l_y = [line2line[0][1], line2line[1][1]]
            ax.plot(l2l_x, l2l_y, 'm-', linewidth=2, alpha=0.8, linestyle='--', label='Line2Line', zorder=2)
            ax.scatter(line2line[0][0], line2line[0][1], c='magenta', s=80, marker='o', edgecolor='black', zorder=5)
            ax.scatter(line2line[1][0], line2line[1][1], c='magenta', s=80, marker='s', edgecolor='black', zorder=5, label='Exit point')
        
        pad_deg_lon = max(radius_deg_lon * 1.5, 0.05)
        pad_deg_lat = max(radius_deg_lat * 1.5, 0.05)
        
        ax.axis('equal')
        ax.set_xlim(cx - pad_deg_lon, cx + pad_deg_lon)
        ax.set_ylim(cy - pad_deg_lat, cy + pad_deg_lat)
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_title(f'Search Step {step_num}: Node {current_node} | Radius: {max_distance:.1f}m')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.show()
        
        return fig, ax
    
    def plot_transect(self, transect_nodes, transect_line, path_nodes, radius=500):
        """Plot the transect with nodes and circles"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if self.point_coords:
            xs = [x for x, y in self.point_coords.values()]
            ys = [y for x, y in self.point_coords.values()]
            ax.scatter(xs, ys, c='lightgray', s=5, alpha=0.5, label='Mesh nodes')
        
        if transect_line:
            lx = [p[0] for p in transect_line]
            ly = [p[1] for p in transect_line]
            ax.plot(lx, ly, 'b-', linewidth=2, label='Transect line')
        
        if path_nodes:
            px = [self.point_coords[n][0] for n in path_nodes]
            py = [self.point_coords[n][1] for n in path_nodes]
            ax.scatter(px, py, c='red', s=50, zorder=5, label='Selected nodes')
            ax.plot(px, py, 'r-', linewidth=2, zorder=4, label='Path')
            
            for i, n in enumerate(path_nodes):
                x, y = self.point_coords[n]
                ax.annotate(str(i+1), (x, y), fontsize=8, xytext=(5, 5), textcoords='offset points')
        
        if transect_line:
            start = transect_line[0]
            end = transect_line[-1]
            ax.scatter(start[0], start[1], c='blue', s=100, marker='s', label='Line start', zorder=10)
            ax.scatter(end[0], end[1], c='blue', s=100, marker='*', label='Line end', zorder=10)
        
        if path_nodes:
            sx, sy = self.point_coords[path_nodes[0]]
            ax.scatter(sx, sy, c='green', s=100, marker='o', edgecolor='black', linewidth=2, label='Start node', zorder=10)
        
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Transect Extraction Results')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def extract_transect(self, grd_file, shapefile_path, start_point_id=0):
        """Main function: extract transect from GRD and shapefile"""
        print(f"Reading GRD: {grd_file}")
        if not self.read_grd(grd_file):
            return
        
        print(f"Reading shapefile: {shapefile_path}")
        result = self.read_shapefile(shapefile_path)
        if result is None:
            return
        
        transect_line, start_id = result
        
        if start_point_id != 0:
            start_id = start_point_id
        
        print("Extracting transect path...")
        start_pos = transect_line[0]
        end_pos = transect_line[-1]
        
        # Pass ALL required parameters
        path = self.build_path_along_transect(
            start_point_id=start_id,
            end_pos=end_pos,
            start_pos=start_pos,
            transect_line=transect_line,
            point_features=self.points  # Pass the GRD points
        )
        
        if not path or len(path) < 2:
            print("No path found!")
            return
        
        print(f"Extracted path with {len(path)} nodes")
        self.plot_transect(path, transect_line)
        
        print("\nSelected nodes:")
        for i, (node_id, x, y) in enumerate(path):
            print(f"  {i+1:3d}: Node {node_id:6d}  ({x:.6f}, {y:.6f})")
        
        return path

    def build_path_along_transect(self, start_point_id, end_pos, start_pos, transect_line, point_features):
        """
        Build path using:
        1. Buffer: 500m each side of transect line
        2. Circle: max_distance = furthest connected node
        3. Line2line: angle from last inside point to first outside point on transect
        4. Selection: node with closest angle to line2line angle
        5. End detection: when close to transect end, snap to end node
        """
        def azimuth(x1, y1, x2, y2):
            return math.atan2(y2 - y1, x2 - x1)
        
        # ============================================================
        # STEP 1: Convert point_features to dictionary with coordinates
        # ============================================================
        if isinstance(point_features, dict):
            point_coords = {}
            for pid, (x, y, _, _) in point_features.items():
                point_coords[pid] = (x, y)
        elif isinstance(point_features, list):
            point_coords = {}
            for i, (x, y) in enumerate(point_features):
                point_coords[i + 1] = (x, y)
        else:
            print("ERROR: point_features is not a dict or list")
            return []
        
        print(f"Using {len(point_coords)} points from GRD")
        
        # ============================================================
        # STEP 2: Store transect and create dense version
        # ============================================================
        original_transect = transect_line
        
        # Calculate total transect length
        total_transect_length = 0
        for i in range(len(transect_line) - 1):
            lx1, ly1 = transect_line[i]
            lx2, ly2 = transect_line[i + 1]
            total_transect_length += self._calculate_distance(lx1, ly1, lx2, ly2)
        print(f"Total transect length: {total_transect_length/1000:.1f} km")
        
        # Get end point of transect (for end detection)
        transect_end_point = transect_line[-1]
        transect_start_point = transect_line[0]
        
        # Divide transect into 1000 points for buffer calculations
        dense_transect = []
        if len(transect_line) > 1:
            total_segments = len(transect_line) - 1
            points_per_segment = max(1, 1000 // total_segments)
            for i in range(total_segments):
                lx1, ly1 = transect_line[i]
                lx2, ly2 = transect_line[i + 1]
                for j in range(points_per_segment + 1):
                    t = j / points_per_segment
                    dense_transect.append((lx1 + t * (lx2 - lx1), ly1 + t * (ly2 - ly1)))
        else:
            dense_transect = transect_line
        
        BUFFER = 2000  # 2000m each side (increased from 500)
        DIRECT = 50    # 50m direct selection
        END_THRESHOLD = 500  # 500m from transect end - snap to end node
        
        # ============================================================
        # STEP 3: Find end node (closest to end_pos)
        # ============================================================
        end_node = None
        min_dist = float('inf')
        for node_id, (x, y) in point_coords.items():
            dist = self._calculate_distance(x, y, end_pos[0], end_pos[1])
            if dist < min_dist:
                min_dist = dist
                end_node = node_id
        
        if end_node is None:
            print("No end node found")
            return []
        
        print(f"End node: {end_node} (distance to transect end: {min_dist:.1f}m)")
        
        # ============================================================
        # STEP 4: Find start node
        # ============================================================
        if start_point_id == 0 or start_point_id not in point_coords:
            min_dist = float('inf')
            for node_id, (x, y) in point_coords.items():
                dist = self._calculate_distance(x, y, start_pos[0], start_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    start_point_id = node_id
            print(f"Auto-detected start node: {start_point_id}")
        
        # ============================================================
        # STEP 5: Initialize path and tracking variables
        # ============================================================
        visited = set([start_point_id])
        path = [(start_point_id, point_coords[start_point_id][0], point_coords[start_point_id][1])]
        current_node = start_point_id
        path_length = 0.0
        step_counter = 0
        
        # Track position along transect to prevent going backwards
        transect_position = 0
        
        print(f"\n{'='*70}")
        print("BUILDING TRANSECT PATH")
        print("="*70)
        print(f"Buffer: {BUFFER}m each side, End threshold: {END_THRESHOLD}m")
        print("="*70)
        
        # ============================================================
        # STEP 6: Main loop
        # ============================================================
        while current_node != end_node:
            step_counter += 1
            cx, cy = point_coords[current_node]
            
            print(f"\n{'─'*70}")
            print(f"STEP {step_counter}: Current node {current_node}")
            print(f"  Position: ({cx:.6f}, {cy:.6f})")
            
            # ============================================================
            # END DETECTION: Check if we're close to transect end
            # ============================================================
            dist_to_transect_end = self._calculate_distance(cx, cy, transect_end_point[0], transect_end_point[1])
            dist_to_end_node = self._calculate_distance(cx, cy, point_coords[end_node][0], point_coords[end_node][1])
            
            print(f"  Distance to transect end: {dist_to_transect_end:.1f}m")
            print(f"  Distance to end node: {dist_to_end_node:.1f}m")
            
            # If close to transect end, snap to end node
            if dist_to_transect_end < END_THRESHOLD or dist_to_end_node < END_THRESHOLD:
                print(f"  ✅ Close to transect end! Snapping to end node: {end_node}")
                # Add end node to path
                ex, ey = point_coords[end_node]
                step_distance = self._calculate_distance(cx, cy, ex, ey)
                path_length += step_distance
                path.append((end_node, ex, ey))
                visited.add(end_node)
                current_node = end_node
                break
            
            # Check if reached end (original threshold)
            dist_to_end = self._calculate_distance(cx, cy, end_pos[0], end_pos[1])
            if dist_to_end < 50:
                print(f"  ✅ Reached end! Distance: {dist_to_end:.1f}m")
                break
            
            # Get connected nodes (excluding visited)
            connected = self.node_adjacency.get(current_node, [])
            unvisited = [n for n in connected if n not in visited]
            print(f"  Connected: {len(connected)}, Unvisited: {len(unvisited)}")
            
            if not unvisited:
                if len(path) > 1:
                    print(f"  ⚠️ Backtracking from {current_node}")
                    path.pop()
                    current_node = path[-1][0]
                    continue
                break
            
            # ============================================================
            # STEP 7: Filter nodes within buffer
            # ============================================================
            buffered_nodes = []
            for node_id in unvisited:
                if node_id not in point_coords:
                    continue
                nx, ny = point_coords[node_id]
                min_dist_to_line = min(self._calculate_distance(nx, ny, lx, ly) 
                                      for lx, ly in dense_transect)
                if min_dist_to_line <= BUFFER:
                    buffered_nodes.append((node_id, min_dist_to_line, nx, ny))
            
            print(f"  Nodes in buffer ({BUFFER}m): {len(buffered_nodes)}")
            
            if not buffered_nodes:
                closest = min(unvisited, key=lambda n: 
                    self._calculate_distance(cx, cy, point_coords[n][0], point_coords[n][1]))
                selected = closest
                max_dist = 0
                line2line = None
                local_azimuth = 0
                print(f"  ⚠️ No nodes in buffer, using closest: {selected}")
            else:
                # ============================================================
                # STEP 8: Compute max_distance (furthest connected node)
                # ============================================================
                max_dist = max(self._calculate_distance(cx, cy, nx, ny) 
                              for _, _, nx, ny in buffered_nodes)
                print(f"  Circle radius: {max_dist:.1f}m")
                
                # ============================================================
                # STEP 9: Find transect line exit point (FORWARD ONLY)
                # ============================================================
                
                # Find closest point on transect starting from current position
                closest_idx = transect_position
                closest_dist = float('inf')
                for i in range(transect_position, len(transect_line)):
                    lx, ly = transect_line[i]
                    dist = self._calculate_distance(cx, cy, lx, ly)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = i
                
                # Update transect position
                transect_position = closest_idx
                
                # Find inside and outside points
                inside_point = None
                outside_point = None
                
                for i in range(transect_position, len(transect_line)):
                    lx, ly = transect_line[i]
                    dist = self._calculate_distance(cx, cy, lx, ly)
                    
                    if dist <= max_dist:
                        inside_point = (lx, ly)
                    else:
                        outside_point = (lx, ly)
                        transect_position = i  # Update for next step
                        break
                
                # If we never exited the circle, use the last point
                if outside_point is None and inside_point is not None:
                    outside_point = inside_point
                    transect_position = len(transect_line) - 1
                
                # If we never found an inside point, use current position
                if inside_point is None:
                    inside_point = (cx, cy)
                if outside_point is None:
                    outside_point = (cx, cy)
                
                # ============================================================
                # STEP 10: Compute line2line angle
                # ============================================================
                line2line_angle = math.atan2(outside_point[1] - inside_point[1],
                                            outside_point[0] - inside_point[0])
                line2line = [inside_point, outside_point]
                
                print(f"  Line2line:")
                print(f"    Inside: ({inside_point[0]:.6f}, {inside_point[1]:.6f})")
                print(f"    Outside: ({outside_point[0]:.6f}, {outside_point[1]:.6f})")
                print(f"    Angle: {math.degrees(line2line_angle):.1f}°")
                
                # ============================================================
                # STEP 11: Evaluate candidates
                # ============================================================
                print(f"\n  {'─'*50}")
                print(f"  {'Node ID':>8} | {'Dist to curr':>12} | {'Angle':>8} | {'Angle diff':>10} | {'In circle':>8}")
                print(f"  {'─'*50}")
                
                candidates_inside = []
                for node_id, _, nx, ny in buffered_nodes:
                    dist_to_current = self._calculate_distance(cx, cy, nx, ny)
                    node_angle = math.atan2(ny - cy, nx - cx)
                    angle_diff = abs(node_angle - line2line_angle)
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    angle_diff_deg = math.degrees(angle_diff)
                    in_circle = dist_to_current <= max_dist
                    
                    print(f"  {node_id:>8} | {dist_to_current:>12.1f}m | {math.degrees(node_angle):>8.1f}° | {angle_diff_deg:>10.1f}° | {str(in_circle):>8}")
                    
                    if in_circle:
                        candidates_inside.append((node_id, angle_diff, dist_to_current))
                
                print(f"  {'─'*50}")
                print(f"  Candidates inside circle: {len(candidates_inside)}")
                
                # ============================================================
                # STEP 12: Select best node
                # ============================================================
                if candidates_inside:
                    # Select node with smallest angle difference, then smallest distance
                    best_node = min(candidates_inside, key=lambda x: (x[1], x[2]))[0]
                    best_angle_diff = min(c[1] for c in candidates_inside)
                    print(f"\n  ✅ Selected: {best_node} (angle diff: {math.degrees(best_angle_diff):.1f}°)")
                    selected = best_node
                else:
                    # Fallback: closest node in buffer
                    selected = min(buffered_nodes, key=lambda n: 
                        self._calculate_distance(cx, cy, n[2], n[3]))[0]
                    print(f"\n  ⚠️ No nodes inside circle, using closest in buffer: {selected}")
                
                local_azimuth = line2line_angle
            
            # ============================================================
            # STEP 13: Plot and add to path
            # ============================================================
            self._plot_search_step(
                current_node=current_node,
                unvisited=unvisited,
                max_distance=max_dist if buffered_nodes else 0,
                azimuth_line=local_azimuth if buffered_nodes else 0,
                best_node=selected,
                fallback_node=None,
                transect_line=original_transect,
                current_transect=path,
                step_num=step_counter,
                line2line=line2line if buffered_nodes else None
            )
            
            sx, sy = point_coords[selected]
            step_distance = self._calculate_distance(cx, cy, sx, sy)
            path_length += step_distance
            
            path.append((selected, sx, sy))
            visited.add(selected)
            current_node = selected
            
            print(f"\n  → Moved to {selected} (step: {step_distance:.1f}m)")
            print(f"  → Total path: {path_length/1000:.2f} km")
        
        # ============================================================
        # STEP 14: Final summary
        # ============================================================
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"  Total nodes: {len(path)}")
        print(f"  Path distance: {path_length/1000:.2f} km")
        print(f"  Transect length: {total_transect_length/1000:.1f} km")
        print(f"  Ratio: {path_length/total_transect_length:.3f}")
        print("="*70)
        
        return path


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    GRD_FILE = "/home/utente/Documenti/OMBRES/grid_ff/adri_lags_15mPiles_276714_excluded.grd"
    SHAPEFILE = "/home/utente/Documenti/OMBRES/QGIS/line_along_adriatic.shp"
    
    extractor = TransectExtractor()
    path = extractor.extract_transect(GRD_FILE, SHAPEFILE, start_point_id=0)
    
    if path:
        print(f"\n✅ Transect extraction complete! Found {len(path)} nodes.")