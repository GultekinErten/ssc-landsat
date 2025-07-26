"""
Topographic Feature Extractor (Single Raster Version)

This script extracts pixel values from one or more raster layers for a list of input point coordinates.
Each raster layer must fully cover the spatial extent of the input points.
Useful for extracting topographic features like slope, aspect, SPI, etc.

Usage:
-------
python extract_topo_features_single.py \
    --input_csv path/to/input.csv \
    --raster_dir path/to/raster_folder \
    --output_csv path/to/output.csv

Notes:
- Input CSV must contain 'lat' and 'lon' columns (EPSG:4326).
- Raster layers should be named like slope.tif, aspect.tif, spi.tif, etc.
- All rasters must be in the same projection (e.g., EPSG:3395).

Author: Gültekin Erten  
License: MIT
"""

import os
import argparse
import pandas as pd
import rasterio
from pyproj import Transformer

# ------------------- Argument Parser -------------------
parser = argparse.ArgumentParser(description="Extract topographic features from raster layers for input coordinates.")
parser.add_argument("--input_csv", required=True, help="Path to input CSV containing lat/lon columns.")
parser.add_argument("--raster_dir", required=True, help="Path to directory containing raster layers.")
parser.add_argument("--output_csv", required=True, help="Path to save the output CSV with extracted values.")
args = parser.parse_args()

input_csv = args.input_csv
raster_dir = args.raster_dir
output_csv = args.output_csv

# ------------------- Configuration -------------------
layers = [
    "slope", "aspect", "twi", "spi", "flow_acc", "flow_dir",
    "plan_curv", "prof_curv", "streams", "valley_depth", "filled"
]

# Coordinate transformer: EPSG:4326 → EPSG:3395
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)

# ------------------- Read Input -------------------
df = pd.read_csv(input_csv)
points = df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)

# ------------------- Feature Extraction -------------------
output_data = []

for _, row in points.iterrows():
    lat, lon = row.lat, row.lon
    x, y = transformer.transform(lon, lat)

    result = {"lat": lat, "lon": lon}

    for layer in layers:
        filename = f"{layer}.tif"
        raster_path = os.path.join(raster_dir, filename)

        if not os.path.exists(raster_path):
            result[layer] = None
            continue

        try:
            with rasterio.open(raster_path) as ds:
                row_pix, col_pix = ds.index(x, y)
                value = ds.read(1)[row_pix, col_pix]
                result[layer] = None if value == ds.nodata else value
        except Exception:
            result[layer] = None

    output_data.append(result)

# ------------------- Save Output -------------------
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_csv, index=False)
print(f"[✓] Finished. Output saved to: {output_csv}")
