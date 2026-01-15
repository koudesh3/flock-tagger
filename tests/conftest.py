"""Shared test fixtures for flock-tagger core tests."""

import pytest


@pytest.fixture
def sample_cameras():
    """Sample camera data for testing."""
    return [
        {"id": "cam1", "lat": 39.7392, "lon": -104.9903},  # Denver
        {"id": "cam2", "lat": 39.7400, "lon": -104.9900},  # ~90m from cam1
        {"id": "cam3", "lat": 40.0150, "lon": -105.2705},  # Boulder
        {"id": "cam4", "lat": 38.8339, "lon": -104.8214},  # Colorado Springs
    ]


@pytest.fixture
def invalid_cameras():
    """Cameras with missing lat/lon fields."""
    return [
        {"id": "valid", "lat": 39.7392, "lon": -104.9903},
        {"id": "no_lat", "lon": -104.9903},
        {"id": "no_lon", "lat": 39.7392},
        {"id": "no_coords"},
    ]


@pytest.fixture
def simple_bbox():
    """Standard test bounding box (min_lon, min_lat, max_lon, max_lat)."""
    return (0.0, 0.0, 1.0, 1.0)
