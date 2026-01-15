"""Main pipeline: region filtering, tile generation, parallel processing, and result merging."""

import argparse
import json
import os
import shutil
import sys

from flock_tagger.camera import load_cameras, build_camera_grid
from flock_tagger.osm import get_osm_bbox
from flock_tagger.tiling import (
    generate_tile_grid,
    process_all_tiles,
    merge_tiles
)


def cleanup_tmp():
    """Clean up temporary working directory."""
    try:
        shutil.rmtree('data/tmp')
        print("Cleaned up data/tmp/ directory")
    except Exception as e:
        print(f"Warning: Could not clean up data/tmp/ directory: {e}", file=sys.stderr)


def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(
        description='Tile-based parallel OSM tagger',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with large tiles (faster)
  %(prog)s data/input/colorado-latest.osm.pbf data/output/colorado-tagged.osm.pbf --tile-size 1.0 --workers 4

  # Full processing with 0.1 degree tiles
  %(prog)s data/input/colorado-latest.osm.pbf data/output/colorado-tagged.osm.pbf --workers 8
        """
    )
    parser.add_argument('input_pbf', help='Input OSM PBF file')
    parser.add_argument('output_pbf', help='Output tagged PBF file')
    parser.add_argument('-w', '--workers', type=int, default=8, help='Number of parallel workers (default: 8)')
    parser.add_argument('--tile-size', type=float, default=0.1, help='Tile size in degrees (default: 0.1)')
    parser.add_argument('--bbox', help='Bounding box as "min_lon,min_lat,max_lon,max_lat" (default: auto-detect from input file)')
    parser.add_argument('--keep-tmp', action='store_true', help='Keep data/tmp/ directory after completion')

    args = parser.parse_args()

    try:
        # Step 1: Filter cameras to region
        print("Filtering cameras to region...")
        cameras = load_cameras('data/input/cameras.json')

        # Parse bbox if provided, otherwise auto-detect from input file
        if args.bbox:
            min_lon, min_lat, max_lon, max_lat = map(float, args.bbox.split(','))
            region_bbox = (min_lon, min_lat, max_lon, max_lat)
        else:
            # Auto-detect bounding box from input file
            print("Auto-detecting bounding box from input file...")
            region_bbox = get_osm_bbox(args.input_pbf)
            print(f"Detected bbox: {region_bbox}")

        # Filter cameras with 0.5 degree buffer
        buffer = 0.5
        region_cameras = [c for c in cameras
                          if (region_bbox[1] - buffer) <= c['lat'] <= (region_bbox[3] + buffer)
                          and (region_bbox[0] - buffer) <= c['lon'] <= (region_bbox[2] + buffer)]

        cameras_json = 'data/tmp/cameras_region.json'
        os.makedirs('data/tmp', exist_ok=True)
        with open(cameras_json, 'w') as f:
            json.dump(region_cameras, f)
        print(f"Filtered to {len(region_cameras)} cameras (from {len(cameras)} global)")

        # Step 2: Build camera grid for spatial optimization
        # Use same grid size as tiles for efficient pre-filtering
        print(f"Building camera spatial grid (cell_size={args.tile_size}°)...")
        camera_grid, cell_size = build_camera_grid(region_cameras, cell_size=args.tile_size)
        print(f"Grid contains {len(camera_grid)} occupied cells")

        # Step 3: Generate tile grid
        print(f"Calculating tile grid (tile_size={args.tile_size}°)...")
        tiles = generate_tile_grid(region_bbox, tile_size=args.tile_size)
        print(f"Grid contains {len(tiles)} tile boundaries")

        # Step 4: Process tiles in parallel
        tagged_pbfs, node_stats, tile_stats = process_all_tiles(
            tiles,
            args.input_pbf,
            camera_grid,
            cell_size,
            max_workers=args.workers
        )

        if not tagged_pbfs:
            print("ERROR: No tiles were successfully tagged!", file=sys.stderr)
            sys.exit(1)

        # Step 5: Merge results
        print("Merging tiles...")
        merge_tiles(tagged_pbfs, args.output_pbf)

        # Step 5.5: Build and save statistics
        total_tagged = (node_stats.nodes_level_1 +
                        node_stats.nodes_level_2 +
                        node_stats.nodes_level_3)

        pipeline_stats = {
            'cameras': {
                'total_global': len(cameras),
                'filtered_to_region': len(region_cameras)
            },
            'tiles': {
                'total_generated': tile_stats.total_tiles,
                'with_cameras': tile_stats.tiles_with_cameras,
                'non_empty': tile_stats.non_empty_tiles,
                'processed_successfully': tile_stats.processed_successfully,
                'errors': tile_stats.errors,
                'skipped': tile_stats.skipped
            },
            'nodes': {
                'total_scanned': node_stats.total_nodes,
                'total_tagged': total_tagged,
                'tagged_level_1': node_stats.nodes_level_1,
                'tagged_level_2': node_stats.nodes_level_2,
                'tagged_level_3': node_stats.nodes_level_3,
                'untagged': node_stats.total_nodes - total_tagged
            },
            'processing': {
                'workers': args.workers,
                'tile_size_degrees': args.tile_size
            }
        }

        # Derive stats filename from output filename
        output_dir = os.path.dirname(args.output_pbf)
        output_base = os.path.splitext(os.path.basename(args.output_pbf))[0]
        stats_path = os.path.join(output_dir, f"{output_base}-stats.json")

        with open(stats_path, 'w') as f:
            json.dump(pipeline_stats, f, indent=2)

        print(f"\nComplete! Tagged output written to: {args.output_pbf}")
        print(f"  Statistics written to: {stats_path}")
        print(f"\nSummary:")
        print(f"  Cameras: {len(region_cameras):,} (filtered from {len(cameras):,} global)")
        print(f"  Tiles: {len(tagged_pbfs)}/{tile_stats.total_tiles} processed")
        print(f"  Nodes scanned: {node_stats.total_nodes:,}")
        print(f"  Nodes tagged: {total_tagged:,}")
        print(f"    Level 1 (100-250m): {node_stats.nodes_level_1:,}")
        print(f"    Level 2 (50-100m): {node_stats.nodes_level_2:,}")
        print(f"    Level 3 (0-50m): {node_stats.nodes_level_3:,}")

        # Step 6: Cleanup tmp directory
        if not args.keep_tmp:
            print("\nCleaning up...")
            cleanup_tmp()

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        print("Keeping data/tmp/ directory for resume (use --keep-tmp to control cleanup after completion)")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
