"""OSM file processing: single-pass streaming with haversine distance calculation."""

from math import radians, cos, sin, sqrt, atan2
import os
import subprocess
from typing import List, Dict, Optional, Tuple

import osmium
from shapely.geometry import Point, box
from shapely.strtree import STRtree

from .camera import get_surveillance_level


EARTH_RADIUS_M = 6371000
CAMERA_RANGE_M = 250
CAMERA_RANGE_DEG = 0.00225  # ~250m at equator (conservative for all latitudes)


def get_osm_bbox(input_path: str) -> Tuple[float, float, float, float]:
    """Extract bounding box from OSM PBF file.

    Tries to read from header first (fast), falls back to calculating from data if unavailable.

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    # Try fast path: read from header
    try:
        result = subprocess.run(
            ['osmium', 'fileinfo', '-g', 'header.box', input_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        bbox_str = result.stdout.strip().strip('()')
        if bbox_str and ',' in bbox_str:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox_str.split(','))
            return (min_lon, min_lat, max_lon, max_lat)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        # Header doesn't have bbox or osmium command failed
        pass

    # Fallback: calculate bbox from data (slow, but uses fast C++ implementation)
    try:
        result = subprocess.run(
            ['osmium', 'fileinfo', '--extended', '-g', 'data.bbox', input_path],
            capture_output=True,
            text=True,
            check=True
        )
        bbox_str = result.stdout.strip().strip('()')
        if bbox_str and ',' in bbox_str:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox_str.split(','))
            return (min_lon, min_lat, max_lon, max_lat)
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to extract bounding box from {input_path}: {e}")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two lat/lon points."""
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError(f"Latitude must be between -90 and 90: got {lat1}, {lat2}")
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError(f"Longitude must be between -180 and 180: got {lon1}, {lon2}")

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_M * c


class TaggingHandler(osmium.SimpleHandler):
    """Single-pass OSM handler that tags nodes with surveillance levels."""

    def __init__(self, cameras: List[Dict], output_writer: osmium.SimpleWriter):
        super().__init__()
        self._validate_cameras(cameras)
        self.cameras = cameras
        self.writer = output_writer
        self.tagged_count = 0
        self.stats = {
            'nodes_level_1': 0,
            'nodes_level_2': 0,
            'nodes_level_3': 0,
            'total_nodes': 0
        }

        self.camera_tree = None
        self.camera_points = []
        self.camera_point_to_data = {}

        if cameras:
            self._build_spatial_index(cameras)

    def _validate_cameras(self, cameras: List[Dict]) -> None:
        """Validate camera data structure."""
        if cameras is None:
            raise ValueError("Cameras list cannot be None")

        for i, cam in enumerate(cameras):
            if not isinstance(cam, dict):
                raise ValueError(f"Camera at index {i} must be a dict, got {type(cam)}")
            if 'lat' in cam or 'lon' in cam:
                if 'lat' not in cam or 'lon' not in cam:
                    raise ValueError(f"Camera at index {i} has incomplete coordinates")
                if not isinstance(cam['lat'], (int, float)) or not isinstance(cam['lon'], (int, float)):
                    raise ValueError(f"Camera at index {i} has non-numeric coordinates")

    def _build_spatial_index(self, cameras: List[Dict]) -> None:
        """Build R-tree spatial index for efficient camera queries."""
        camera_points = [
            Point(cam['lon'], cam['lat']) 
            for cam in cameras 
            if 'lat' in cam and 'lon' in cam
        ]

        if not camera_points:
            return

        self.camera_tree = STRtree(camera_points)
        self.camera_points = camera_points
        
        # Use (lon, lat) tuples as keys for reliable cross-process lookup
        # (avoids PYTHONHASHSEED issues with Point objects as keys)
        self.camera_point_to_data = {
            (cam['lon'], cam['lat']): cam
            for cam in cameras 
            if 'lat' in cam and 'lon' in cam
        }

    def node(self, n):
        """Process node: calculate distance to cameras and tag if within range."""
        self.stats['total_nodes'] += 1

        if not n.location.valid():
            self.writer.add_node(n)
            return

        node_lat, node_lon = n.location.lat, n.location.lon

        max_level = self._calculate_surveillance_level(node_lat, node_lon)

        if max_level:
            self._tag_and_write_node(n, max_level)
            self.tagged_count += 1
        else:
            self.writer.add(n)

    def _calculate_surveillance_level(self, node_lat: float, node_lon: float) -> Optional[int]:
        """Return highest surveillance level for node based on nearby cameras."""
        if self.camera_tree is not None:
            return self._spatial_index_lookup(node_lat, node_lon)
        else:
            return self._linear_scan_lookup(node_lat, node_lon)

    def _spatial_index_lookup(self, node_lat: float, node_lon: float) -> Optional[int]:
        """Query R-tree for nearby cameras and return highest surveillance level."""
        search_box = box(
            node_lon - CAMERA_RANGE_DEG,
            node_lat - CAMERA_RANGE_DEG,
            node_lon + CAMERA_RANGE_DEG,
            node_lat + CAMERA_RANGE_DEG
        )

        nearby_indices = self.camera_tree.query(search_box)
        
        max_level = None
        min_distance = float('inf')

        for idx in nearby_indices:
            cam_point = self.camera_points[idx]
            cam = self.camera_point_to_data.get((cam_point.x, cam_point.y))
            
            if not cam:
                continue

            distance = haversine_distance(node_lat, node_lon, cam['lat'], cam['lon'])

            if distance <= 50:
                return 3  # Early termination: highest level found

            if distance < min_distance:
                min_distance = distance
                level = get_surveillance_level(distance)
                if level and (max_level is None or level > max_level):
                    max_level = level

        return max_level

    def _linear_scan_lookup(self, node_lat: float, node_lon: float) -> Optional[int]:
        """Fallback: scan all cameras linearly (used when spatial index unavailable)."""
        max_level = None
        min_distance = float('inf')

        for cam in self.cameras:
            if 'lat' not in cam or 'lon' not in cam:
                continue

            distance = haversine_distance(node_lat, node_lon, cam['lat'], cam['lon'])

            if distance <= 50:
                return 3

            if distance < min_distance:
                min_distance = distance
                level = get_surveillance_level(distance)
                if level and (max_level is None or level > max_level):
                    max_level = level

        return max_level

    def _tag_and_write_node(self, node, level: int) -> None:
        """Add surveillance tag to node and write to output."""
        tags = dict(node.tags)
        tags['surveillance'] = str(level)
        modified_node = node.replace(tags=tags)
        self.writer.add(modified_node)

        if level == 1:
            self.stats['nodes_level_1'] += 1
        elif level == 2:
            self.stats['nodes_level_2'] += 1
        elif level == 3:
            self.stats['nodes_level_3'] += 1

    def way(self, w):
        self.writer.add(w)

    def relation(self, r):
        self.writer.add(r)


def process_osm_file(
    input_path: str,
    output_path: str,
    cameras: List[Dict],
    verbose: bool = False
) -> Dict[str, int]:
    """Process OSM file and tag nodes with surveillance levels (single-pass)."""
    if not input_path:
        raise ValueError("Input path cannot be empty")
    if not output_path:
        raise ValueError("Output path cannot be empty")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.isfile(input_path):
        raise ValueError(f"Input path is not a file: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    if output_dir and not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}")

    if cameras is None:
        raise ValueError("Cameras list cannot be None")

    if verbose:
        print(f"Processing {input_path} with {len(cameras)} cameras...")

    try:
        with osmium.SimpleWriter(output_path, overwrite=True) as writer:
            handler = TaggingHandler(cameras, writer)
            handler.apply_file(input_path)
    except Exception as e:
        raise RuntimeError(f"Failed to process OSM file: {e}") from e

    if verbose:
        total_tagged = sum(
            handler.stats[f'nodes_level_{i}']
            for i in range(1, 4)
        )
        print(f"Tagged {total_tagged:,} nodes:")
        print(f"  Level 3 (≤50m): {handler.stats['nodes_level_3']:,}")
        print(f"  Level 2 (≤100m): {handler.stats['nodes_level_2']:,}")
        print(f"  Level 1 (≤250m): {handler.stats['nodes_level_1']:,}")

    return handler.stats