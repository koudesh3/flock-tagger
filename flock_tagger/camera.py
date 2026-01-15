"""Camera operations: loading, spatial grid, and surveillance level calculation."""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# Type aliases
CameraGrid = Dict[Tuple[int, int], List[Dict]]
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

# Constants
CAMERA_RANGE_M = 250
METERS_PER_DEGREE = 111000  # Approximate at equator


@dataclass
class Camera:
    """Camera with latitude and longitude."""
    lat: float
    lon: float


def load_cameras(json_path: str) -> List[Dict]:
    """Load camera data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            cameras = json.load(f)

        if not cameras:
            print(f"Warning: No cameras found in {json_path}", file=sys.stderr)

        return cameras
    except FileNotFoundError:
        print(f"Error: Camera file '{json_path}' not found", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{json_path}': {e}", file=sys.stderr)
        raise


def build_camera_grid(cameras: List[Dict], cell_size: float = 0.1) -> Tuple[CameraGrid, float]:
    """
    Build spatial grid for camera lookup optimization.

    Grid cells should match tile size for efficient tile filtering before extraction.
    """
    if not isinstance(cameras, list):
        raise TypeError("cameras must be a list")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    grid = defaultdict(list)
    for camera in cameras:
        if 'lat' not in camera or 'lon' not in camera:
            continue

        lat, lon = camera['lat'], camera['lon']
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue

        cell_x = int(lon / cell_size)
        cell_y = int(lat / cell_size)
        grid[(cell_x, cell_y)].append(camera)

    return grid, cell_size


def get_cameras_for_tile(
    grid: CameraGrid,
    cell_size: float,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float
) -> List[Dict]:
    """Get cameras relevant to a tile, including 1-cell buffer for 250m range."""
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("Invalid bounds: min coordinates must be less than max")

    min_cell_x = int(min_lon / cell_size) - 1
    max_cell_x = int(max_lon / cell_size) + 1
    min_cell_y = int(min_lat / cell_size) - 1
    max_cell_y = int(max_lat / cell_size) + 1

    cameras = []
    for cell_x in range(min_cell_x, max_cell_x + 1):
        for cell_y in range(min_cell_y, max_cell_y + 1):
            cameras.extend(grid.get((cell_x, cell_y), []))

    return cameras


def tile_has_cameras(
    cameras: List[Dict],
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float
) -> bool:
    """Check if any camera is within 250m of tile bounds."""
    if not isinstance(cameras, list):
        raise TypeError("cameras must be a list")
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("Invalid bounds: min coordinates must be less than max")

    buffer_deg = CAMERA_RANGE_M / METERS_PER_DEGREE

    for camera in cameras:
        if 'lat' not in camera or 'lon' not in camera:
            continue

        lat, lon = camera['lat'], camera['lon']
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue

        if (min_lon - buffer_deg <= lon <= max_lon + buffer_deg and
            min_lat - buffer_deg <= lat <= max_lat + buffer_deg):
            return True

    return False


def get_surveillance_level(distance_m: float) -> Optional[int]:
    """
    Convert distance to surveillance level.

    Returns:
        3 if distance ≤ 50m (highest surveillance)
        2 if distance ≤ 100m (medium surveillance)
        1 if distance ≤ 250m (lowest surveillance)
        None if distance > 250m
    """
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")

    if distance_m <= 50:
        return 3
    elif distance_m <= 100:
        return 2
    elif distance_m <= 250:
        return 1
    else:
        return None