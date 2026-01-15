"""Core behavior tests for flock-tagger critical algorithms."""

import pytest
import subprocess
import tempfile
import os
from unittest.mock import patch, MagicMock

from flock_tagger.camera import build_camera_grid, get_surveillance_level
from flock_tagger.osm import haversine_distance, TaggingHandler, get_osm_bbox
from flock_tagger.tiling import generate_tile_grid
from shapely.geometry import Point, box


class TestHaversineDistance:
    """Test distance calculation accuracy - core algorithm."""

    def test_calculates_accurate_known_distances(self):
        """
        Behavior: Returns accurate distance for known coordinate pairs.

        Arrange: North Pole to South Pole (half Earth's circumference)
        Act: Calculate haversine distance
        Assert: Distance equals π × R (≈20,015,087m) within ±100m
        """
        north_pole_lat, north_pole_lon = 90.0, 0.0
        south_pole_lat, south_pole_lon = -90.0, 0.0

        distance = haversine_distance(north_pole_lat, north_pole_lon, south_pole_lat, south_pole_lon)

        # Half Earth's circumference: π × R where R = 6,371,000m
        # Expected: π × 6,371,000 ≈ 20,015,087 meters
        expected = 20_015_087
        assert expected - 100 <= distance <= expected + 100

    def test_returns_zero_for_identical_coordinates(self):
        """
        Behavior: Returns exactly 0 for same point.

        Arrange: Same lat/lon pair
        Act: Calculate distance
        Assert: Returns 0.0
        """
        lat, lon = 39.7392, -104.9903

        distance = haversine_distance(lat, lon, lat, lon)

        assert distance == 0.0


class TestGetSurveillanceLevel:
    """Test surveillance level threshold logic - core business logic."""

    def test_threshold_boundary_values(self):
        """
        Behavior: Maps distances to correct levels at boundary values.

        Arrange: Boundary distances (0, 50, 51, 100, 101, 250, 251)
        Act: Get surveillance level for each
        Assert: Correct level returned per specification
        """
        test_cases = [
            (0, 3),
            (50, 3),
            (51, 2),
            (100, 2),
            (101, 1),
            (250, 1),
            (251, None),
        ]

        for distance, expected in test_cases:
            level = get_surveillance_level(distance)
            assert level == expected, f"Distance {distance}m should return {expected}"


class TestBuildCameraGrid:
    """Test spatial indexing correctness - critical for performance."""

    def test_cameras_indexed_to_correct_grid_cells(self, sample_cameras):
        """
        Behavior: Cameras placed in mathematically correct grid cells.

        Arrange: Cameras with known lat/lon values
        Act: Build grid with cell_size=0.1
        Assert: Each camera in grid matches calculated cell position
        """
        cell_size = 0.1
        grid, returned_size = build_camera_grid(sample_cameras, cell_size=cell_size)

        assert returned_size == cell_size

        for cell_coord, cameras in grid.items():
            cell_x, cell_y = cell_coord
            for camera in cameras:
                expected_x = int(camera['lon'] / cell_size)
                expected_y = int(camera['lat'] / cell_size)
                assert (expected_x, expected_y) == cell_coord

    def test_skips_cameras_without_coordinates(self, invalid_cameras):
        """
        Behavior: Safely ignores cameras missing lat/lon.

        Arrange: Mix of valid/invalid cameras
        Act: Build grid
        Assert: Only valid cameras indexed
        """
        grid, _ = build_camera_grid(invalid_cameras, cell_size=0.1)

        total_in_grid = sum(len(cameras) for cameras in grid.values())

        assert total_in_grid == 1


class TestGenerateTileGrid:
    """Test tile grid generation - ensures complete coverage."""

    def test_tiles_within_bbox_bounds(self, simple_bbox):
        """
        Behavior: All generated tiles stay within bbox boundaries.

        Arrange: Simple bbox (0, 0, 1, 1)
        Act: Generate tile grid
        Assert: Every tile within bounds
        """
        min_lon, min_lat, max_lon, max_lat = simple_bbox
        tile_size = 0.1

        tiles = generate_tile_grid(simple_bbox, tile_size=tile_size)

        assert len(tiles) > 0

        for tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat in tiles:
            assert tile_min_lon >= min_lon
            assert tile_min_lat >= min_lat
            assert tile_max_lon <= max_lon
            assert tile_max_lat <= max_lat

    def test_complete_coverage_no_gaps(self):
        """
        Behavior: Tiles cover entire bbox without gaps.

        Arrange: Evenly divisible bbox (0, 0, 1, 1) with tile_size=0.1
        Act: Generate tiles
        Assert: Correct number of tiles (10x10=100), complete coverage
        """
        bbox = (0.0, 0.0, 1.0, 1.0)
        tile_size = 0.1

        tiles = generate_tile_grid(bbox, tile_size=tile_size)

        assert len(tiles) == 100

        total_area = sum(
            (tile_max_lon - tile_min_lon) * (tile_max_lat - tile_min_lat)
            for tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat in tiles
        )
        bbox_area = 1.0

        assert abs(total_area - bbox_area) < 0.0001


class TestTaggingHandlerCameraLookup:
    """Test camera lookup in TaggingHandler - regression tests for coordinate-based lookup."""

    def test_can_lookup_all_cameras_by_coordinates(self, sample_cameras):
        """
        Behavior: All cameras can be looked up using (lon, lat) tuple keys.

        Regression test: After the fix, we should be able to lookup any camera
        using its (lon, lat) coordinates as a tuple key.

        Arrange: TaggingHandler with sample cameras
        Act: Try to lookup each camera by its coordinates
        Assert: All cameras can be found
        """
        handler = TaggingHandler(sample_cameras, output_writer=None)

        # After the fix, camera_point_to_data should use (lon, lat) tuples as keys
        for cam in sample_cameras:
            key = (cam['lon'], cam['lat'])
            looked_up_cam = handler.camera_point_to_data.get(key)

            assert looked_up_cam is not None, (
                f"Failed to lookup camera at ({cam['lon']}, {cam['lat']}). "
                f"Dictionary should use (lon, lat) tuples as keys, not Point objects."
            )
            assert looked_up_cam['id'] == cam['id']

    def test_strtree_indices_map_to_cameras(self, sample_cameras):
        """
        Behavior: STRtree query returns indices that map to correct cameras.

        Regression test: Shapely 2.x STRtree.query() returns indices (integers),
        not Point objects. We need to use these indices to get camera coordinates,
        then lookup via (lon, lat) tuple.

        Arrange: TaggingHandler with sample cameras
        Act: Query STRtree and use indices to lookup cameras
        Assert: All lookups succeed using the index->coordinates->dict pattern
        """
        handler = TaggingHandler(sample_cameras, output_writer=None)

        # Query near Denver cameras
        search_box = box(-104.9913, 39.7382, -104.9893, 39.7402)
        nearby_indices = handler.camera_tree.query(search_box)

        # Should find at least 2 Denver cameras
        assert len(nearby_indices) >= 1, "Should find cameras near Denver"

        # After the fix: use indices to get Point objects, extract coordinates, lookup in dict
        # Build list of Point objects in same order as STRtree
        camera_points = [Point(cam['lon'], cam['lat']) for cam in sample_cameras if 'lat' in cam and 'lon' in cam]

        found_cameras = []
        for idx in nearby_indices:
            # Get the Point object at this index
            cam_point = camera_points[idx]

            # Use coordinates as tuple key (THE FIX)
            key = (cam_point.x, cam_point.y)
            cam_data = handler.camera_point_to_data.get(key)

            assert cam_data is not None, (
                f"Failed to lookup camera at index {idx} with coordinates ({cam_point.x}, {cam_point.y}). "
                f"After fix, this should work using (lon, lat) tuple keys."
            )
            found_cameras.append(cam_data)

        assert len(found_cameras) >= 1, "Should successfully lookup at least one camera"


class TestGetOsmBbox:
    """Test bounding box extraction from OSM files."""

    @patch('flock_tagger.osm.subprocess.run')
    def test_extracts_bbox_from_header_when_available(self, mock_run):
        """
        Behavior: Extracts bbox from file header when available (fast path).

        Arrange: Mock osmium fileinfo to return bbox from header
        Act: Call get_osm_bbox()
        Assert: Returns correct bbox tuple from header
        """
        # Mock successful osmium fileinfo call
        mock_result = MagicMock()
        mock_result.stdout = "(-109.0631,36.56774,-100.4637,41.00403)"
        mock_run.return_value = mock_result

        bbox = get_osm_bbox('test.osm.pbf')

        # Should return tuple of (min_lon, min_lat, max_lon, max_lat)
        assert isinstance(bbox, tuple)
        assert len(bbox) == 4
        assert bbox == (-109.0631, 36.56774, -100.4637, 41.00403)

        # Verify it used the fast path
        mock_run.assert_called_once()

    @patch('flock_tagger.osm.subprocess.run')
    def test_calculates_bbox_when_header_unavailable(self, mock_run):
        """
        Behavior: Falls back to calculating bbox when header unavailable.

        Arrange: Mock header read to fail, mock extended data.bbox read to succeed
        Act: Call get_osm_bbox()
        Assert: Falls back to osmium --extended and returns bbox
        """
        # First call (header.box) fails, second call (data.bbox with --extended) succeeds
        mock_result = MagicMock()
        mock_result.stdout = "(-120.0,35.0,-115.0,40.0)"

        mock_run.side_effect = [
            subprocess.CalledProcessError(1, 'osmium'),  # header.box fails
            mock_result  # data.bbox succeeds
        ]

        bbox = get_osm_bbox('test.osm.pbf')

        # Should return bbox from extended calculation
        assert bbox == (-120.0, 35.0, -115.0, 40.0)

        # Verify it tried fast path first, then fell back to extended
        assert mock_run.call_count == 2
        # Second call should use --extended flag
        assert '--extended' in mock_run.call_args_list[1][0][0]
