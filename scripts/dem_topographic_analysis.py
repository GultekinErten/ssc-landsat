"""
DEM Topographic Analysis using WhiteboxTools
--------------------------------------------

This script performs automated topographic analysis on DEM tiles using WhiteboxTools.
You can specify input raster name(s) and an output directory via command-line arguments.

Usage Example:
--------------
python dem_topographic_analysis.py --raster_names SRTM_bottom_left SRTM_bottom_right --output_dir D:/outputs

Author: Gültekin Erten
License: MIT
"""

import os
import argparse
from whitebox import WhiteboxTools

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="DEM Topographic Analysis using WhiteboxTools")
parser.add_argument(
    "--raster_names",
    nargs="+",
    required=True,
    help="List of input raster names without .tif extension (e.g., SRTM_bottom_left SRTM_bottom_right)"
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to output directory"
)
args = parser.parse_args()

raster_names = args.raster_names
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# --- Initialize WhiteboxTools ---
wbt = WhiteboxTools()
wbt.set_working_dir(output_dir)

# --- Process Each Raster ---
for name in raster_names:
    input_raster = f"{name}.tif"
    input_raster_path = os.path.join(output_dir, input_raster)

    filled = os.path.join(output_dir, f"{name}_filled.tif")
    flow_dir = os.path.join(output_dir, f"flow_dir_{name}.tif")
    flow_acc = os.path.join(output_dir, f"flow_acc_{name}.tif")

    print(f" Processing: {input_raster}")

    wbt.fill_depressions_planchon_and_darboux(input_raster_path, filled)
    wbt.d8_pointer(filled, flow_dir)
    wbt.d8_flow_accumulation(filled, flow_acc, out_type="sca")

    wbt.wetness_index(filled, flow_acc, os.path.join(output_dir, f"twi_{name}.tif"))
    wbt.stream_power_index(filled, flow_acc, os.path.join(output_dir, f"spi_{name}.tif"))

    wbt.slope(filled, os.path.join(output_dir, f"slope_{name}.tif"))
    wbt.aspect(filled, os.path.join(output_dir, f"aspect_{name}.tif"))
    wbt.plan_curvature(filled, os.path.join(output_dir, f"plan_curv_{name}.tif"))
    wbt.profile_curvature(filled, os.path.join(output_dir, f"prof_curv_{name}.tif"))

    wbt.extract_streams(flow_acc, os.path.join(output_dir, f"streams_{name}.tif"), threshold=1000)
    wbt.relative_topographic_position(filled, os.path.join(output_dir, f"valley_depth_{name}.tif"))

    print(f" Completed: {name}\n")
