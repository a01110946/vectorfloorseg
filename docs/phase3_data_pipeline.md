# VectorFloorSeg Implementation - Phase 3: Data Preparation Pipeline

## Overview

This phase implements the complete data preparation pipeline for VectorFloorSeg, including SVG processing, graph construction, and dataset creation. This is the core component that converts vector floorplans into the dual graph representations required by the model.

## Prerequisites

- Completed Phase 1: Environment Setup
- Completed Phase 2: Project Structure Setup
- Active virtual environment: `vectorfloorseg_env`

## 3.1 SVG Processing and Graph Construction

```python
# File: src/data/svg_processor.py
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch_geometric.data import Data, HeteroData
import cv2
from pathlib import Path
import logging
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

class SVGFloorplanProcessor:
    """Process SVG floorplans into graph representations for VectorFloorSeg."""
    
    def __init__(
        self, 
        image_size: Tuple[int, int] = (256, 256),
        line_extension_length: float = 50.0,
        min_region_area: float = 100.0
    ):
        self.image_size = image_size
        self.line_extension_length = line_extension_length
        self.min_region_area = min_region_area
        self.logger = logging.getLogger(__name__)
        
    def parse_svg(self, svg_path: str) -> Dict[str, Any]:
        """Parse SVG file and extract line segments."""
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid SVG file: {e}")
        
        # Extract line segments from various SVG elements
        lines = []
        
        # Handle different SVG elements
        for element in root.iter():
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            
            if tag == 'line':
                line = self._parse_line(element)
                if line:
                    lines.append(line)
            elif tag == 'path':
                path_lines = self._parse_path(element)
                lines.extend(path_lines)
            elif tag == 'polyline':
                polyline_lines = self._parse_polyline(element)
                lines.extend(polyline_lines)
            elif tag == 'polygon':
                polygon_lines = self._parse_polygon(element)
                lines.extend(polygon_lines)
            elif tag == 'rect':
                rect_lines = self._parse_rect(element)
                lines.extend(rect_lines)
        
        # Get SVG dimensions and viewBox
        viewbox = self._get_viewbox(root)
        dimensions = self._get_dimensions(root)
        
        return {
            'lines': lines,
            'viewBox': viewbox,
            'dimensions': dimensions,
            'num_lines': len(lines)
        }
    
    def _parse_line(self, element) -> Optional[Tuple[float, float, float, float]]:
        """Parse a line element into coordinates."""
        try:
            x1 = float(element.get('x1', 0))
            y1 = float(element.get('y1', 0))
            x2 = float(element.get('x2', 0))
            y2 = float(element.get('y2', 0))
            
            # Skip zero-length lines
            if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
                return None
                
            return (x1, y1, x2, y2)
        except (ValueError, TypeError):
            return None
    
    def _parse_path(self, element) -> List[Tuple[float, float, float, float]]:
        """Parse path elements into line segments."""
        d = element.get('d', '').strip()
        if not d:
            return []
        
        lines = []
        
        # Simple path parsing - handles M, L, H, V, Z commands
        # For production, consider using svg.path library
        commands = self._tokenize_path(d)
        current_x, current_y = 0.0, 0.0
        start_x, start_y = 0.0, 0.0
        
        i = 0
        while i < len(commands):
            cmd = commands[i].upper()
            
            if cmd == 'M':  # Move to
                if i + 2 < len(commands):
                    current_x = float(commands[i + 1])
                    current_y = float(commands[i + 2])
                    start_x, start_y = current_x, current_y
                    i += 3
                else:
                    break
                    
            elif cmd == 'L':  # Line to
                if i + 2 < len(commands):
                    new_x = float(commands[i + 1])
                    new_y = float(commands[i + 2])
                    lines.append((current_x, current_y, new_x, new_y))
                    current_x, current_y = new_x, new_y
                    i += 3
                else:
                    break
                    
            elif cmd == 'H':  # Horizontal line
                if i + 1 < len(commands):
                    new_x = float(commands[i + 1])
                    lines.append((current_x, current_y, new_x, current_y))
                    current_x = new_x
                    i += 2
                else:
                    break
                    
            elif cmd == 'V':  # Vertical line
                if i + 1 < len(commands):
                    new_y = float(commands[i + 1])
                    lines.append((current_x, current_y, current_x, new_y))
                    current_y = new_y
                    i += 2
                else:
                    break
                    
            elif cmd == 'Z':  # Close path
                lines.append((current_x, current_y, start_x, start_y))
                current_x, current_y = start_x, start_y
                i += 1
            else:
                i += 1
        
        return lines
    
    def _tokenize_path(self, path_data: str) -> List[str]:
        """Tokenize SVG path data."""
        import re
        # Split on commands and coordinates
        tokens = re.findall(r'[MLHVCSQTAZ]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', path_data)
        return tokens
    
    def _parse_polyline(self, element) -> List[Tuple[float, float, float, float]]:
        """Parse polyline into individual line segments."""
        points_str = element.get('points', '').strip()
        if not points_str:
            return []
        
        # Parse points
        coords = points_str.replace(',', ' ').split()
        points = []
        
        for i in range(0, len(coords) - 1, 2):
            try:
                x = float(coords[i])
                y = float(coords[i + 1])
                points.append((x, y))
            except (ValueError, IndexError):
                continue
        
        # Convert to line segments
        lines = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            lines.append((x1, y1, x2, y2))
        
        return lines
    
    def _parse_polygon(self, element) -> List[Tuple[float, float, float, float]]:
        """Parse polygon into line segments."""
        lines = self._parse_polyline(element)
        
        # Close the polygon if it has points
        if lines:
            # Get first and last points
            first_line = lines[0]
            last_line = lines[-1]
            
            # Add closing line if not already closed
            if abs(first_line[0] - last_line[2]) > 1e-6 or abs(first_line[1] - last_line[3]) > 1e-6:
                lines.append((last_line[2], last_line[3], first_line[0], first_line[1]))
        
        return lines
    
    def _parse_rect(self, element) -> List[Tuple[float, float, float, float]]:
        """Parse rectangle into four line segments."""
        try:
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            width = float(element.get('width', 0))
            height = float(element.get('height', 0))
            
            # Four sides of rectangle
            return [
                (x, y, x + width, y),         # Top
                (x + width, y, x + width, y + height),  # Right
                (x + width, y + height, x, y + height), # Bottom
                (x, y + height, x, y)         # Left
            ]
        except (ValueError, TypeError):
            return []
    
    def _get_viewbox(self, root) -> Optional[Tuple[float, float, float, float]]:
        """Extract viewBox from SVG root."""
        viewbox = root.get('viewBox')
        if viewbox:
            try:
                coords = [float(x) for x in viewbox.split()]
                if len(coords) == 4:
                    return tuple(coords)
            except ValueError:
                pass
        return None
    
    def _get_dimensions(self, root) -> Dict[str, float]:
        """Extract width and height from SVG root."""
        width_str = root.get('width', '256')
        height_str = root.get('height', '256')
        
        try:
            # Remove units (px, pt, etc.)
            width = float(''.join(c for c in width_str if c.isdigit() or c == '.'))
            height = float(''.join(c for c in height_str if c.isdigit() or c == '.'))
        except ValueError:
            width, height = 256.0, 256.0
        
        return {'width': width, 'height': height}
    
    def normalize_coordinates(
        self, 
        lines: List[Tuple], 
        dimensions: Dict,
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple]:
        """Normalize coordinates to target size range."""
        if not lines:
            return lines
        
        if target_size is None:
            target_size = self.image_size
        
        # Find bounding box of all coordinates
        all_x = []
        all_y = []
        for line in lines:
            all_x.extend([line[0], line[2]])
            all_y.extend([line[1], line[3]])
        
        if not all_x or not all_y:
            return lines
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Calculate scale and offset to fit target size
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return lines
        
        scale_x = (target_size[0] - 20) / width  # Leave 10px margin on each side
        scale_y = (target_size[1] - 20) / height
        scale = min(scale_x, scale_y)  # Maintain aspect ratio
        
        # Center the normalized coordinates
        offset_x = (target_size[0] - width * scale) / 2
        offset_y = (target_size[1] - height * scale) / 2
        
        normalized_lines = []
        for x1, y1, x2, y2 in lines:
            nx1 = (x1 - min_x) * scale + offset_x
            ny1 = (y1 - min_y) * scale + offset_y
            nx2 = (x2 - min_x) * scale + offset_x
            ny2 = (y2 - min_y) * scale + offset_y
            normalized_lines.append((nx1, ny1, nx2, ny2))
        
        return normalized_lines
    
    def extend_lines_and_partition(self, lines: List[Tuple]) -> Dict[str, Any]:
        """Extend lines to create closed regions using computational geometry."""
        if not lines:
            return {'original_lines': [], 'extended_lines': [], 'regions': []}
        
        # Convert lines to LineString objects
        line_strings = []
        for x1, y1, x2, y2 in lines:
            line_strings.append(LineString([(x1, y1), (x2, y2)]))
        
        # Extend lines
        extended_lines = self._extend_lines(lines)
        
        # Find intersections and create planar subdivision
        regions = self._detect_regions_advanced(extended_lines)
        
        return {
            'original_lines': lines,
            'extended_lines': extended_lines,
            'regions': regions,
            'num_regions': len(regions)
        }
    
    def _extend_lines(self, lines: List[Tuple]) -> List[Tuple]:
        """Extend lines to image boundaries or until intersection."""
        extended_lines = lines.copy()
        
        for x1, y1, x2, y2 in lines:
            # Calculate line direction vector
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            
            if length < 1e-6:  # Skip near-zero length lines
                continue
            
            # Normalize direction
            dx /= length
            dy /= length
            
            # Extend in both directions
            extend_length = self.line_extension_length
            
            # Extension from start point
            ext_x1 = x1 - dx * extend_length
            ext_y1 = y1 - dy * extend_length
            
            # Extension from end point
            ext_x2 = x2 + dx * extend_length
            ext_y2 = y2 + dy * extend_length
            
            # Clip to image boundaries
            ext_x1 = max(0, min(self.image_size[0], ext_x1))
            ext_y1 = max(0, min(self.image_size[1], ext_y1))
            ext_x2 = max(0, min(self.image_size[0], ext_x2))
            ext_y2 = max(0, min(self.image_size[1], ext_y2))
            
            # Add extended line if it's different from original
            if (abs(ext_x1 - x1) > 1 or abs(ext_y1 - y1) > 1 or 
                abs(ext_x2 - x2) > 1 or abs(ext_y2 - y2) > 1):
                extended_lines.append((ext_x1, ext_y1, ext_x2, ext_y2))
        
        return extended_lines
    
    def _detect_regions_advanced(self, lines: List[Tuple]) -> List[List[Tuple]]:
        """Advanced region detection using computational geometry."""
        if not lines:
            return []
        
        # Create a simple grid-based approach for now
        # In production, implement proper planar subdivision
        regions = self._grid_based_region_detection(lines)
        
        # Filter out regions that are too small
        filtered_regions = []
        for region in regions:
            if len(region) >= 3:
                area = self._polygon_area(region)
                if area >= self.min_region_area:
                    filtered_regions.append(region)
        
        return filtered_regions
    
    def _grid_based_region_detection(self, lines: List[Tuple]) -> List[List[Tuple]]:
        """Simple grid-based region detection."""
        # Create a grid and identify connected components
        grid_size = 32
        grid = np.zeros((grid_size, grid_size))
        
        # Rasterize lines onto grid
        for x1, y1, x2, y2 in lines:
            # Convert to grid coordinates
            gx1 = int(x1 * grid_size / self.image_size[0])
            gy1 = int(y1 * grid_size / self.image_size[1])
            gx2 = int(x2 * grid_size / self.image_size[0])
            gy2 = int(y2 * grid_size / self.image_size[1])
            
            # Draw line on grid using Bresenham-like algorithm
            self._draw_line_on_grid(grid, gx1, gy1, gx2, gy2)
        
        # Find connected components in the inverted grid
        from scipy.ndimage import label
        inverted_grid = 1 - grid
        labeled_grid, num_regions = label(inverted_grid)
        
        regions = []
        for region_id in range(1, num_regions + 1):
            # Find boundary of this region
            region_mask = (labeled_grid == region_id)
            if np.sum(region_mask) < 4:  # Skip very small regions
                continue
            
            # Convert back to image coordinates
            boundary = self._extract_region_boundary(region_mask, grid_size)
            if boundary:
                regions.append(boundary)
        
        return regions
    
    def _draw_line_on_grid(self, grid: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """Draw line on grid using simple line drawing."""
        h, w = grid.shape
        
        # Clip coordinates
        x1, y1 = max(0, min(w-1, x1)), max(0, min(h-1, y1))
        x2, y2 = max(0, min(w-1, x2)), max(0, min(h-1, y2))
        
        # Simple line drawing
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx == 0 and dy == 0:
            grid[y1, x1] = 1
            return
        
        steps = max(dx, dy)
        x_step = (x2 - x1) / steps if steps > 0 else 0
        y_step = (y2 - y1) / steps if steps > 0 else 0
        
        for i in range(steps + 1):
            x = int(x1 + i * x_step)
            y = int(y1 + i * y_step)
            if 0 <= x < w and 0 <= y < h:
                grid[y, x] = 1
    
    def _extract_region_boundary(self, region_mask: np.ndarray, grid_size: int) -> List[Tuple]:
        """Extract boundary of a region and convert to image coordinates."""
        # Find contours of the region
        region_uint8 = (region_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Take the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert to image coordinates
        boundary = []
        scale_x = self.image_size[0] / grid_size
        scale_y = self.image_size[1] / grid_size
        
        for point in largest_contour:
            x, y = point[0]
            img_x = x * scale_x
            img_y = y * scale_y
            boundary.append((img_x, img_y))
        
        return boundary
    
    def build_primal_graph(self, lines: List[Tuple]) -> Data:
        """Build primal graph where vertices are line endpoints and edges are lines."""
        if not lines:
            return Data(
                x=torch.zeros((0, 66)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 128)),
                pos=torch.zeros((0, 2))
            )
        
        # Extract unique vertices (line endpoints)
        vertices = set()
        for x1, y1, x2, y2 in lines:
            vertices.add((x1, y1))
            vertices.add((x2, y2))
        
        # Create vertex mapping
        vertex_list = list(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
        
        # Build edges and features
        edges = []
        edge_features = []
        
        for x1, y1, x2, y2 in lines:
            v1_idx = vertex_to_idx[(x1, y1)]
            v2_idx = vertex_to_idx[(x2, y2)]
            
            # Add bidirectional edges
            edges.extend([[v1_idx, v2_idx], [v2_idx, v1_idx]])
            
            # Calculate edge features
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            angle = np.arctan2(dy, dx)
            
            # Edge features: geometric properties
            edge_feat = [
                np.cos(angle), np.sin(angle),  # Direction
                length / 255.0,                # Normalized length
                1.0,                          # Is original line (vs extended)
                x1 / 255.0, y1 / 255.0,      # Start coordinates
                x2 / 255.0, y2 / 255.0       # End coordinates
            ]
            
            # Pad to fixed size (128 dimensions)
            edge_feat.extend([0.0] * (128 - len(edge_feat)))
            edge_features.extend([edge_feat, edge_feat])  # For both directions
        
        # Convert to tensors
        vertex_coords = torch.tensor(vertex_list, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.zeros((0, 128))
        
        # Vertex features: coordinates + positional encoding
        pos_encoding = self._positional_encoding(vertex_coords, dim=64)
        vertex_features = torch.cat([vertex_coords, pos_encoding], dim=1) if len(vertex_coords) > 0 else torch.zeros((0, 66))
        
        return Data(
            x=vertex_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=vertex_coords
        )
    
    def build_dual_graph(self, regions: List[List[Tuple]]) -> Data:
        """Build dual graph where vertices are regions and edges are adjacencies."""
        if not regions:
            return Data(
                x=torch.zeros((0, 66)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 128)),
                pos=torch.zeros((0, 2))
            )
        
        # Calculate region features
        region_features = []
        region_centroids = []
        
        for region in regions:
            if len(region) >= 3:
                # Calculate centroid
                centroid_x = sum(p[0] for p in region) / len(region)
                centroid_y = sum(p[1] for p in region) / len(region)
                region_centroids.append((centroid_x, centroid_y))
                
                # Calculate geometric features
                area = self._polygon_area(region)
                perimeter = self._polygon_perimeter(region)
                
                # Basic region features
                region_feat = [
                    centroid_x / 255.0, centroid_y / 255.0,  # Normalized centroid
                    area / (255*255),                        # Normalized area
                    perimeter / (4*255),                     # Normalized perimeter
                ]
                region_features.append(region_feat)
        
        # Build adjacency edges
        edges = []
        edge_features = []
        
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if self._regions_adjacent(regions[i], regions[j]):
                    edges.extend([[i, j], [j, i]])  # Bidirectional
                    
                    # Edge features: relationship between regions
                    centroid_i = region_centroids[i]
                    centroid_j = region_centroids[j]
                    
                    distance = np.sqrt((centroid_i[0] - centroid_j[0])**2 + 
                                     (centroid_i[1] - centroid_j[1])**2)
                    
                    edge_feat = [distance / 255.0, 0.0]  # Distance and placeholder
                    edge_feat.extend([0.0] * (128 - len(edge_feat)))
                    edge_features.extend([edge_feat, edge_feat])
        
        # Convert to tensors
        if region_features:
            vertex_coords = torch.tensor(region_centroids, dtype=torch.float)
            vertex_features_basic = torch.tensor(region_features, dtype=torch.float)
            
            # Add positional encoding
            pos_encoding = self._positional_encoding(vertex_coords, dim=62)
            vertex_features = torch.cat([vertex_features_basic, pos_encoding], dim=1)
        else:
            vertex_features = torch.zeros((0, 66))
            vertex_coords = torch.zeros((0, 2))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.zeros((0, 128))
        
        return Data(
            x=vertex_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=vertex_coords
        )
    
    def _positional_encoding(self, coords: torch.Tensor, dim: int = 64) -> torch.Tensor:
        """Generate sinusoidal positional encoding for coordinates."""
        if coords.size(0) == 0:
            return torch.zeros((0, dim))
        
        pe = torch.zeros(coords.size(0), dim)
        
        # Generate frequency scales
        div_term = torch.exp(torch.arange(0, dim//2, dtype=torch.float) * 
                           -(np.log(10000.0) / (dim//2)))
        
        # Apply to x coordinates
        pe[:, 0::4] = torch.sin(coords[:, 0:1] * div_term[None, :dim//4])
        pe[:, 1::4] = torch.cos(coords[:, 0:1] * div_term[None, :dim//4])
        
        # Apply to y coordinates
        if dim > 2:
            pe[:, 2::4] = torch.sin(coords[:, 1:2] * div_term[None, :dim//4])
            pe[:, 3::4] = torch.cos(coords[:, 1:2] * div_term[None, :dim//4])
        
        return pe
    
    def _polygon_area(self, vertices: List[Tuple]) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        
        area = 0.0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        
        return abs(area) / 2.0
    
    def _polygon_perimeter(self, vertices: List[Tuple]) -> float:
        """Calculate polygon perimeter."""
        if len(vertices) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            dx = vertices[j][0] - vertices[i][0]
            dy = vertices[j][1] - vertices[i][1]
            perimeter += np.sqrt(dx*dx + dy*dy)
        
        return perimeter
    
    def _regions_adjacent(self, region1: List[Tuple], region2: List[Tuple]) -> bool:
        """Check if two regions share a boundary."""
        try:
            # Convert to Shapely polygons
            poly1 = Polygon(region1)
            poly2 = Polygon(region2)
            
            # Check if they touch (share boundary)
            return poly1.touches(poly2) or poly1.intersects(poly2)
        except:
            # Fallback to distance-based check
            centroid1 = (sum(p[0] for p in region1) / len(region1),
                        sum(p[1] for p in region1) / len(region1))
            centroid2 = (sum(p[0] for p in region2) / len(region2),
                        sum(p[1] for p in region2) / len(region2))
            
            distance = np.sqrt((centroid1[0] - centroid2[0])**2 + 
                             (centroid1[1] - centroid2[1])**2)
            
            return distance < 50  # Threshold for adjacency
    
    def process_svg_to_graphs(self, svg_path: str) -> Dict[str, Any]:
        """Complete pipeline to convert SVG to primal and dual graphs."""
        try:
            # Parse SVG
            svg_data = self.parse_svg(svg_path)
            self.logger.info(f"Parsed SVG with {svg_data['num_lines']} lines")
            
            # Normalize coordinates
            normalized_lines = self.normalize_coordinates(
                svg_data['lines'], 
                svg_data['dimensions']
            )
            
            # Extend lines and detect regions
            partition_data = self.extend_lines_and_partition(normalized_lines)
            self.logger.info(f"Detected {partition_data['num_regions']} regions")
            
            # Build graphs
            primal_graph = self.build_primal_graph(partition_data['extended_lines'])
            dual_graph = self.build_dual_graph(partition_data['regions'])
            
            return {
                'primal': primal_graph,
                'dual': dual_graph,
                'lines': normalized_lines,
                'regions': partition_data['regions'],
                'metadata': {
                    'svg_path': svg_path,
                    'num_lines': len(normalized_lines),
                    'num_regions': len(partition_data['regions']),
                    'primal_nodes': primal_graph.x.size(0),
                    'primal_edges': primal_graph.edge_index.size(1),
                    'dual_nodes': dual_graph.x.size(0),
                    'dual_edges': dual_graph.edge_index.size(1)
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing SVG {svg_path}: {e}")
            raise
```

## 3.2 Dataset Classes

```python
# File: src/data/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
import logging
import numpy as np
from PIL import Image
import cv2

from .svg_processor import SVGFloorplanProcessor

class VectorFloorplanDataset(Dataset):
    """Dataset for VectorFloorSeg training and evaluation."""
    
    def __init__(
        self,
        data_root: str,
        dataset_name: str = "R2V",
        split: str = "train",
        transform=None,
        cache_processed: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory containing datasets
            dataset_name: Name of dataset (R2V, CubiCasa-5k)
            split: Data split (train, val, test)
            transform: Data transformation pipeline
            cache_processed: Cache processed graphs to disk
            max_samples: Limit number of samples (for debugging)
        """
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.cache_processed = cache_processed
        self.max_samples = max_samples
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize SVG processor
        self.svg_processor = SVGFloorplanProcessor()
        
        # Dataset-specific settings
        self.room_classes = self._get_room_classes()
        self.num_classes = len(self.room_classes)
        
        # Setup paths
        self.dataset_dir = self.data_root / "raw" / dataset_name
        self.processed_dir = self.data_root / "processed" / dataset_name
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data list
        self.data_list = self._load_data_list()
        
        if max_samples:
            self.data_list = self.data_list[:max_samples]
        
        self.logger.info(f"Loaded {len(self.data_list)} samples from {dataset_name} {split}")
    
    def _get_room_classes(self) -> Dict[str, int]:
        """Get room class mappings for the dataset."""
        if self.dataset_name == "R2V":
            return {
                'background': 0,
                'wall': 1,
                'door': 2,
                'window': 3,
                'washing_room': 4,
                'bedroom': 5,
                'closet': 6,
                'balcony': 7,
                'hall': 8,
                'kitchen': 9,
                'other_room': 10
            }
        elif self.dataset_name == "CubiCasa-5k":
            return {
                'background': 0,
                'bathroom': 1,
                'bedroom': 2,
                'kitchen': 3,
                'living_room': 4,
                'dining_room': 5,
                'hallway': 6,
                'balcony': 7,
                'closet': 8,
                'laundry': 9,
                'entrance': 10,
                'other': 11
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_data_list(self) -> List[Dict]:
        """Load list of data samples."""
        # Try to load from processed metadata
        metadata_file = self.processed_dir / f"{self.split}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        # Generate data list from raw files
        data_list = []
        
        if self.dataset_name == "R2V":
            data_list = self._load_r2v_data_list()
        elif self.dataset_name == "CubiCasa-5k":
            data_list = self._load_cubicasa_data_list()
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(data_list, f, indent=2)
        
        return data_list
    
    def _load_r2v_data_list(self) -> List[Dict]:
        """Load R2V dataset file list."""
        # R2V dataset structure
        split_dir = self.dataset_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"R2V {self.split} directory not found: {split_dir}")
        
        data_list = []
        
        # Find SVG files and corresponding labels
        svg_files = list(split_dir.glob("*.svg"))
        
        for svg_file in svg_files:
            # Look for corresponding label file
            label_file = svg_file.with_suffix('.json')
            
            if label_file.exists():
                data_list.append({
                    'svg_path': str(svg_file),
                    'label_path': str(label_file),
                    'sample_id': svg_file.stem
                })
        
        return data_list
    
    def _load_cubicasa_data_list(self) -> List[Dict]:
        """Load CubiCasa-5k dataset file list."""
        # CubiCasa dataset structure
        split_file = self.dataset_dir / f"{self.split}.txt"
        
        if not split_file.exists():
            raise FileNotFoundError(f"CubiCasa split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            sample_ids = [line.strip() for line in f.readlines()]
        
        data_list = []
        
        for sample_id in sample_ids:
            svg_path = self.dataset_dir / "svg" / f"{sample_id}.svg"
            label_path = self.dataset_dir / "labels" / f"{sample_id}.json"
            
            if svg_path.exists() and label_path.exists():
                data_list.append({
                    'svg_path': str(svg_path),
                    'label_path': str(label_path),
                    'sample_id': sample_id
                })
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        sample_info = self.data_list[idx]
        
        # Try to load from cache
        if self.cache_processed:
            cache_file = self.processed_dir / f"{sample_info['sample_id']}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load cached data for {sample_info['sample_id']}: {e}")
        
        # Process from scratch
        try:
            # Process SVG to graphs
            graphs = self.svg_processor.process_svg_to_graphs(sample_info['svg_path'])
            
            # Load labels
            labels = self._load_labels(sample_info['label_path'], graphs)
            
            # Create sample
            sample = {
                'primal': graphs['primal'],
                'dual': graphs['dual'],
                'boundary_labels': labels['boundary_labels'],
                'room_labels': labels['room_labels'],
                'metadata': {
                    **graphs['metadata'],
                    'sample_id': sample_info['sample_id']
                }
            }
            
            # Apply transforms
            if self.transform:
                sample = self.transform(sample)
            
            # Cache processed data
            if self.cache_processed:
                cache_file = self.processed_dir / f"{sample_info['sample_id']}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(sample, f)
                except Exception as e:
                    self.logger.warning(f"Failed to cache data for {sample_info['sample_id']}: {e}")
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error processing sample {sample_info['sample_id']}: {e}")
            # Return empty sample to avoid breaking training
            return self._get_empty_sample()
    
    def _load_labels(self, label_path: str, graphs: Dict) -> Dict[str, torch.Tensor]:
        """Load ground truth labels for the sample."""
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        # Generate boundary labels (simplified)
        num_edges = graphs['primal'].edge_index.size(1)
        boundary_labels = torch.zeros(num_edges, dtype=torch.long)
        
        # Generate room labels (simplified)
        num_regions = graphs['dual'].x.size(0)
        room_labels = torch.zeros(num_regions, dtype=torch.long)
        
        # TODO: Implement proper label assignment based on ground truth
        # This requires mapping between detected regions and labeled regions
        
        return {
            'boundary_labels': boundary_labels,
            'room_labels': room_labels
        }
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return empty sample for error cases."""
        return {
            'primal': Data(
                x=torch.zeros((1, 66)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 128)),
                pos=torch.zeros((1, 2))
            ),
            'dual': Data(
                x=torch.zeros((1, 66)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 128)),
                pos=torch.zeros((1, 2))
            ),
            'boundary_labels': torch.zeros(0, dtype=torch.long),
            'room_labels': torch.zeros(1, dtype=torch.long),
            'metadata': {'sample_id': 'empty', 'error': True}
        }

def collate_vectorfloorplan(batch: List[Dict]) -> Dict[str, Union[Batch, torch.Tensor]]:
    """Custom collate function for VectorFloorplan data."""
    
    # Separate primal and dual graphs
    primal_graphs = [sample['primal'] for sample in batch]
    dual_graphs = [sample['dual'] for sample in batch]
    
    # Batch graphs
    primal_batch = Batch.from_data_list(primal_graphs)
    dual_batch = Batch.from_data_list(dual_graphs)
    
    # Collect labels
    boundary_labels = []
    room_labels = []
    
    for sample in batch:
        boundary_labels.append(sample['boundary_labels'])
        room_labels.append(sample['room_labels'])
    
    # Concatenate labels
    boundary_labels = torch.cat(boundary_labels, dim=0) if boundary_labels[0].numel() > 0 else torch.zeros(0, dtype=torch.long)
    room_labels = torch.cat(room_labels, dim=0) if room_labels[0].numel() > 0 else torch.zeros(0, dtype=torch.long)
    
    return {
        'primal': primal_batch,
        'dual': dual_batch,
        'boundary_labels': boundary_labels,
        'room_labels': room_labels,
        'metadata': [sample['metadata'] for sample in batch]
    }

def create_dataloader(
    data_root: str,
    dataset_name: str = "R2V",
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for VectorFloorSeg."""
    
    dataset = VectorFloorplanDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split=split,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_vectorfloorplan,
        pin_memory=torch.cuda.is_available()
    )
```

## 3.3 Data Preprocessing Script

```python
# File: preprocess_data.py
"""Data preprocessing script for VectorFloorSeg datasets."""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.datasets import VectorFloorplanDataset, create_dataloader
from src.data.svg_processor import SVGFloorplanProcessor
from src.utils.logging_utils import setup_logging
from src.utils.visualization import FloorplanVisualizer

def preprocess_dataset(
    data_root: str,
    dataset_name: str,
    max_samples_per_split: Optional[int] = None
):
    """Preprocess a complete dataset."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing {dataset_name} dataset")
    
    # Initialize visualizer
    visualizer = FloorplanVisualizer()
    
    splits = ["train", "val", "test"] if dataset_name == "CubiCasa-5k" else ["train", "test"]
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        
        try:
            # Create dataset
            dataset = VectorFloorplanDataset(
                data_root=data_root,
                dataset_name=dataset_name,
                split=split,
                cache_processed=True,
                max_samples=max_samples_per_split
            )
            
            # Process samples
            for i in range(len(dataset)):
                try:
                    sample = dataset[i]
                    
                    if i % 50 == 0:
                        logger.info(f"Processed {i}/{len(dataset)} samples")
                    
                    # Visualize first few samples
                    if i < 5:
                        sample_id = sample['metadata']['sample_id']
                        
                        # Visualize graphs
                        visualizer.plot_primal_graph(
                            sample['primal'].pos,
                            sample['primal'].edge_index,
                            title=f"{dataset_name} {split} - {sample_id} (Primal)",
                            save_name=f"{dataset_name}_{split}_{sample_id}_primal"
                        )
                        
                        # Get regions from metadata if available
                        if 'regions' in sample['metadata']:
                            regions = sample['metadata']['regions']
                            visualizer.plot_dual_graph(
                                regions,
                                room_labels=sample['room_labels'],
                                title=f"{dataset_name} {split} - {sample_id} (Dual)",
                                save_name=f"{dataset_name}_{split}_{sample_id}_dual"
                            )
                
                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    continue
            
            logger.info(f"Completed {split} split: {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}")
            continue
    
    logger.info(f"Dataset {dataset_name} preprocessing complete")

def test_svg_processing():
    """Test SVG processing on sample files."""
    
    logger = logging.getLogger(__name__)
    processor = SVGFloorplanProcessor()
    visualizer = FloorplanVisualizer()
    
    # Create a simple test SVG
    test_svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <svg width="256" height="256" xmlns="http://www.w3.org/2000/svg">
        <line x1="50" y1="50" x2="200" y2="50" stroke="black" stroke-width="2"/>
        <line x1="200" y1="50" x2="200" y2="200" stroke="black" stroke-width="2"/>
        <line x1="200" y1="200" x2="50" y2="200" stroke="black" stroke-width="2"/>
        <line x1="50" y1="200" x2="50" y2="50" stroke="black" stroke-width="2"/>
        <line x1="125" y1="50" x2="125" y2="200" stroke="black" stroke-width="2"/>
    </svg>'''
    
    # Save test SVG
    test_svg_path = Path("test_floorplan.svg")
    with open(test_svg_path, 'w') as f:
        f.write(test_svg_content)
    
    try:
        # Process SVG
        result = processor.process_svg_to_graphs(str(test_svg_path))
        
        logger.info(f"Test SVG processing results:")
        logger.info(f"  Lines: {len(result['lines'])}")
        logger.info(f"  Regions: {len(result['regions'])}")
        logger.info(f"  Primal graph: {result['primal']}")
        logger.info(f"  Dual graph: {result['dual']}")
        
        # Visualize results
        visualizer.plot_primal_graph(
            result['primal'].pos,
            result['primal'].edge_index,
            title="Test SVG - Primal Graph",
            save_name="test_primal"
        )
        
        visualizer.plot_dual_graph(
            result['regions'],
            title="Test SVG - Dual Graph",
            save_name="test_dual"
        )
        
        logger.info("Test SVG processing completed successfully")
        
    except Exception as e:
        logger.error(f"Test SVG processing failed: {e}")
    finally:
        # Clean up
        if test_svg_path.exists():
            test_svg_path.unlink()

def main():
    """Main preprocessing function."""
    
    parser = argparse.ArgumentParser(description='Preprocess VectorFloorSeg datasets')
    parser.add_argument('--data_root', type=str, default='data', help='Root data directory')
    parser.add_argument('--dataset', type=str, choices=['R2V', 'CubiCasa-5k', 'both'], 
                       default='both', help='Dataset to preprocess')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum samples per split (for testing)')
    parser.add_argument('--test_only', action='store_true', 
                       help='Only run SVG processing test')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(experiment_name="preprocessing")
    
    if args.test_only:
        test_svg_processing()
        return
    
    # Preprocess datasets
    datasets = []
    if args.dataset == 'both':
        datasets = ['R2V', 'CubiCasa-5k']
    else:
        datasets = [args.dataset]
    
    for dataset_name in datasets:
        try:
            preprocess_dataset(
                data_root=args.data_root,
                dataset_name=dataset_name,
                max_samples_per_split=args.max_samples
            )
        except Exception as e:
            logger.error(f"Failed to preprocess {dataset_name}: {e}")

if __name__ == "__main__":
    main()
```

## Usage Instructions

### Test SVG Processing
```bash
cd VecFloorSeg
source vectorfloorseg_env/bin/activate

# Test basic SVG processing
python preprocess_data.py --test_only

# Check output visualizations
ls outputs/visualizations/test_*.png
```

### Preprocess Datasets
```bash
# Preprocess R2V dataset (small test)
python preprocess_data.py --dataset R2V --max_samples 10

# Preprocess full datasets
python preprocess_data.py --dataset both
```

### Use in Training
```python
from src.data.datasets import create_dataloader

# Create data loaders
train_loader = create_dataloader(
    data_root="data",
    dataset_name="R2V",
    split="train",
    batch_size=8
)

# Iterate through data
for batch in train_loader:
    primal_data = batch['primal']
    dual_data = batch['dual']
    boundary_labels = batch['boundary_labels']
    room_labels = batch['room_labels']
    break
```

## Next Steps

Proceed to **Phase 4: Model Implementation** to implement the two-stream GAT architecture.

## Troubleshooting

- **SVG parsing errors**: Start with simple SVG files
- **Memory issues**: Reduce max_samples for testing
- **Visualization problems**: Check outputs/visualizations/ directory
- **Dataset loading**: Verify data directory structure matches expected format
