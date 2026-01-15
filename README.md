# Flock OSM Tagger

This tool processes OSM PBF files and tags road nodes with `surveillance:level` (1-3) based on distance to cameras. It uses tile-based parallel processing for efficiently handling large-scale map files.

## Requirements

- Python 3.11+
- Poetry
- osmium-tool (for tile extraction/merging)

## Installation

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install osmium-tool
brew install osmium-tool
```

## Setup

Create the data directories:
```bash
mkdir -p data/input data/output data/tmp
```

Add your files:
- `data/input/cameras.json` - Camera locations (lat/lon)
- `data/input/[region]-latest.osm.pbf` - OSM extract

## Usage

```bash
poetry run python pipeline.py \
  data/input/colorado-latest.osm.pbf \
  data/output/colorado-tagged.osm.pbf \
  --workers 4
```

**Arguments:**
- `input_pbf` - Input OSM PBF file
- `output_pbf` - Output tagged PBF file
- `radius` - Camera buffer radius in meters (e.g., 250)
- `--workers` - Number of parallel workers (default: 8)
- `--tile-size` - Tile size in degrees (default: 0.25)
- `--bbox` - Bounding box as "min_lon,min_lat,max_lon,max_lat"
- `--keep-tmp` - Don't delete temporary files after completion

## Camera JSON Format

```json
[
  {
    "id": 12345,
    "lat": 39.7392,
    "lon": -104.9903
  }
]
```

## Surveillance Levels

- **Level 1**: 0-25m from camera (high coverage)
- **Level 2**: 25-125m from camera (medium coverage)
- **Level 3**: 125-250m from camera (low coverage)

## How It Works

1. **Filter cameras** to region bbox
2. **Generate tile grid** (0.10° tiles)
3. **Extract tiles** in single osmium pass
4. **Process tiles in parallel** - tag nodes based on distance to cameras
5. **Merge results** into final tagged PBF

## Getting OSM Data

I use [Geofabrik](https://download.geofabrik.de/) to get my data!