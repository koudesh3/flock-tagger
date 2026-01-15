"""Tile-based parallel processing: grid generation, extraction, and merging."""

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import List, Tuple, Dict, Optional

from .camera import get_cameras_for_tile, tile_has_cameras
from .osm import process_osm_file


# Type aliases
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
CameraGrid = Dict[Tuple[int, int], List[Dict]]
TileID = int


@dataclass
class TileStats:
    total_tiles: int
    tiles_with_cameras: int
    non_empty_tiles: int
    processed_successfully: int
    errors: int
    skipped: int


@dataclass
class ProcessingStats:
    total_nodes: int = 0
    nodes_level_1: int = 0
    nodes_level_2: int = 0
    nodes_level_3: int = 0
    
    def update(self, other: 'ProcessingStats') -> None:
        self.total_nodes += other.total_nodes
        self.nodes_level_1 += other.nodes_level_1
        self.nodes_level_2 += other.nodes_level_2
        self.nodes_level_3 += other.nodes_level_3


def show_progress(total_tiles: int):
    """Create a thread-safe progress tracking function with colored output."""
    state = {'completed': 0}
    lock = Lock()
    
    GREEN = '\033[92m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    
    def update(increment: int = 1):
        with lock:
            state['completed'] += increment
            
            percent = int((state['completed'] / total_tiles) * 100)
            bar_width = 50
            filled = int(bar_width * state['completed'] / total_tiles)
            
            bar = f"{GREEN}{'█' * filled}{GRAY}{'░' * (bar_width - filled)}{RESET}"
            sys.stdout.write(f"\r[{bar}] {percent}% ({state['completed']}/{total_tiles} tiles)")
            sys.stdout.flush()
    
    # Initialize display
    update(0)
    return update


def generate_tile_grid(bbox: BBox, tile_size: float = 0.1) -> List[BBox]:
    """Generate non-overlapping tiles covering the bounding box."""
    import math
    
    min_lon, min_lat, max_lon, max_lat = bbox
    
    n_lon = math.ceil((max_lon - min_lon) / tile_size)
    n_lat = math.ceil((max_lat - min_lat) / tile_size)
    
    tiles = []
    for i in range(n_lon):
        for j in range(n_lat):
            lon = min_lon + i * tile_size
            lat = min_lat + j * tile_size
            tiles.append((
                lon, lat,
                min(lon + tile_size, max_lon),
                min(lat + tile_size, max_lat)
            ))
    
    return tiles


def extract_tiles(
    tiles: List[BBox],
    tile_ids: List[TileID],
    input_pbf: str,
    batch_size: int = 25
) -> None:
    """Extract tiles from OSM file in batches to avoid memory exhaustion."""
    total = len(tiles)
    
    if total <= batch_size:
        _extract_batch(tiles, tile_ids, input_pbf)
        return
    
    num_batches = (total + batch_size - 1) // batch_size
    _show_extraction_progress(0, num_batches)
    
    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, total)
        
        _extract_batch(
            tiles[start:end],
            tile_ids[start:end],
            input_pbf
        )
        
        _show_extraction_progress(batch_num + 1, num_batches)
    
    print()  # Newline after progress


def _show_extraction_progress(current_batch: int, total_batches: int) -> None:
    """Display extraction progress bar."""
    bar_width = 40
    percent = int((current_batch / total_batches) * 100)
    filled = int(bar_width * current_batch / total_batches)
    bar = f"{'█' * filled}{'░' * (bar_width - filled)}"
    
    sys.stdout.write(f"\r[{bar}] {percent}% (batch {current_batch}/{total_batches})")
    sys.stdout.flush()


def _extract_batch(tiles: List[BBox], tile_ids: List[TileID], input_pbf: str) -> None:
    """Extract a batch of tiles using osmium."""
    config = {
        "extracts": [
            {
                "output": f"data/tmp/tiles/tile_{tile_id}.osm.pbf",
                "bbox": list(bbox)
            }
            for bbox, tile_id in zip(tiles, tile_ids)
        ]
    }
    
    config_path = f"data/tmp/tiles/extract_config_{tile_ids[0]}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    subprocess.run(
        ['osmium', 'extract', '--strategy=simple', '--config', config_path, input_pbf, '--overwrite'],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def filter_tiles_with_cameras(
    tiles: List[BBox],
    camera_grid: CameraGrid,
    cell_size: float
) -> List[TileID]:
    """Return indices of tiles that have camera coverage."""
    tiles_with_cameras = []
    
    for i, (min_lon, min_lat, max_lon, max_lat) in enumerate(tiles):
        tile_cameras = get_cameras_for_tile(
            camera_grid, cell_size,
            min_lat, min_lon, max_lat, max_lon
        )
        
        if tile_has_cameras(tile_cameras, min_lat, min_lon, max_lat, max_lon):
            tiles_with_cameras.append(i)
    
    return tiles_with_cameras


def filter_non_empty_tiles(tile_ids: List[TileID]) -> List[TileID]:
    """Return tile IDs that have actual OSM data (file size > 100 bytes)."""
    return [
        i for i in tile_ids
        if os.path.exists(f'data/tmp/tiles/tile_{i}.osm.pbf')
        and os.path.getsize(f'data/tmp/tiles/tile_{i}.osm.pbf') > 100
    ]


def process_tile(
    tile_id: TileID,
    tile_pbf: str,
    tile_bounds: BBox,
    camera_grid: CameraGrid,
    cell_size: float,
    output_pbf: str
) -> Tuple[TileID, str, Optional[str], Optional[Dict]]:
    """Process a single tile: tag nodes within camera range."""
    try:
        min_lon, min_lat, max_lon, max_lat = tile_bounds

        tile_cameras = get_cameras_for_tile(
            camera_grid, cell_size,
            min_lat, min_lon, max_lat, max_lon
        )

        if not tile_has_cameras(tile_cameras, min_lat, min_lon, max_lat, max_lon):
            return (tile_id, "Skipped - no cameras", None, None)

        stats = process_osm_file(
            tile_pbf,
            output_pbf,
            tile_cameras,
            verbose=False
        )

        total_tagged = stats['nodes_level_1'] + stats['nodes_level_2'] + stats['nodes_level_3']
        stats_str = f"Tagged {total_tagged:,} nodes: L1={stats['nodes_level_1']:,}, L2={stats['nodes_level_2']:,}, L3={stats['nodes_level_3']:,}"

        return (tile_id, stats_str, output_pbf, stats)

    except Exception as e:
        return (tile_id, f"ERROR: {str(e)}", None, None)


def process_all_tiles(
    tiles: List[BBox],
    input_pbf: str,
    camera_grid: CameraGrid,
    cell_size: float,
    max_workers: int = 4
):
    """
    Process tiles in parallel: filter, extract, tag, and aggregate results.

    Returns:
        (tagged_pbfs, processing_stats, tile_stats)
    """
    os.makedirs('data/tmp/tiles', exist_ok=True)
    os.makedirs('data/tmp/tiles/tagged', exist_ok=True)

    # Filter to tiles with camera coverage
    print(f"Filtering {len(tiles)} tiles for camera coverage...")
    tiles_with_cameras = filter_tiles_with_cameras(tiles, camera_grid, cell_size)
    print(f"Found {len(tiles_with_cameras)} tiles with cameras (skipping {len(tiles) - len(tiles_with_cameras)})")

    if not tiles_with_cameras:
        print("No tiles have camera coverage!")
        return ([], ProcessingStats(), TileStats(
            total_tiles=len(tiles),
            tiles_with_cameras=0,
            non_empty_tiles=0,
            processed_successfully=0,
            errors=0,
            skipped=0
        ))

    # Extract tiles
    tiles_exist = all(
        os.path.exists(f'data/tmp/tiles/tile_{i}.osm.pbf')
        for i in tiles_with_cameras
    )

    if tiles_exist:
        print(f"Found existing {len(tiles_with_cameras)} tiles (skipping extraction)")
    else:
        tiles_to_extract = [tiles[i] for i in tiles_with_cameras]
        batch_count = (len(tiles_to_extract) + 24) // 25
        
        if batch_count > 1:
            print(f"Extracting {len(tiles_to_extract)} tiles in {batch_count} batches...")
        else:
            print(f"Extracting {len(tiles_to_extract)} tiles...")
        
        extract_tiles(tiles_to_extract, tiles_with_cameras, input_pbf)

    # Filter to non-empty tiles
    non_empty_tiles = filter_non_empty_tiles(tiles_with_cameras)
    print(f"Found {len(non_empty_tiles)} non-empty tiles")

    # Process tiles in parallel
    print(f"\nProcessing {len(non_empty_tiles)} tiles with {max_workers} workers...")
    update_progress = show_progress(len(non_empty_tiles))

    tagged_pbfs = []
    errors = []
    skipped = 0
    processing_stats = ProcessingStats()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            (
                i,
                executor.submit(
                    process_tile,
                    i,
                    f'data/tmp/tiles/tile_{i}.osm.pbf',
                    tiles[i],
                    camera_grid,
                    cell_size,
                    f'data/tmp/tiles/tagged/tile_{i}_tagged.osm.pbf'
                )
            )
            for i in non_empty_tiles
        ]

        for i, future in futures:
            tile_id, result_msg, output_path, tile_stats = future.result()
            update_progress()

            if "ERROR" in result_msg:
                errors.append(f"Tile {tile_id}: {result_msg}")
            elif "Skipped" in result_msg:
                skipped += 1
            elif output_path:
                tagged_pbfs.append(output_path)
                if tile_stats:
                    tile_processing_stats = ProcessingStats(**tile_stats)
                    processing_stats.update(tile_processing_stats)

    print()  # Newline after progress

    if skipped > 0:
        print(f"Skipped {skipped} tiles without camera coverage")

    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  {error}", file=sys.stderr)

    print(f"Completed processing {len(tagged_pbfs)}/{len(non_empty_tiles)} tiles successfully")

    tile_stats = TileStats(
        total_tiles=len(tiles),
        tiles_with_cameras=len(tiles_with_cameras),
        non_empty_tiles=len(non_empty_tiles),
        processed_successfully=len(tagged_pbfs),
        errors=len(errors),
        skipped=skipped
    )

    return tagged_pbfs, processing_stats, tile_stats


def merge_tiles(tagged_pbfs: List[str], output_pbf: str) -> None:
    """Merge tagged tiles into single output file."""
    subprocess.run(
        ['osmium', 'merge'] + tagged_pbfs + ['-o', output_pbf, '--overwrite'],
        check=True,
        capture_output=True
    )
    print(f"Merged {len(tagged_pbfs)} tiles into {output_pbf}")