#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_ssc.py

This script performs SSC (Suspended Sediment Concentration) prediction using pre-trained CatBoost models.
It accepts satellite images and corresponding DEM files as input, computes relevant features,
filters pixels using QA and NDWI masks, and generates two types of predictions:
1. Direct SSC
2. Exponentiated ln(SSC)

Outputs are saved as GeoTIFFs in specified folders. Script works in both command-line and Spyder environments.

Author: Gültekin Erten
"""

import os
import numpy as np
import rasterio
from rasterio.transform import xy
import pickle
import pandas as pd
from pyproj import Transformer
from datetime import datetime
from tqdm import tqdm
import argparse
from glob import glob

# Suppress numpy warnings like divide by zero
np.seterr(invalid='ignore', divide='ignore')

# ---------------- Helper Functions ---------------- #
def extract_month_from_filename(filename):
    try:
        parts = filename.replace(".tif", "").split("_")
        return int(parts[-2])
    except:
        return None

def get_dem_path(dem_dir, river_name):
    return os.path.join(dem_dir, f"{river_name}_SRTM_DEM.tif")

def decode_qa_pixel(qa_pixel_array):
    return {
        "dilated": ((qa_pixel_array & (1 << 1)) >> 1),
        "cirrus": ((qa_pixel_array & (1 << 2)) >> 2),
        "cloud": ((qa_pixel_array & (1 << 3)) >> 3),
        "shadow": ((qa_pixel_array & (1 << 4)) >> 4)
    }

# ---------------- Main Function ---------------- #
def main(args):
    os.makedirs(args.output_ln, exist_ok=True)
    os.makedirs(args.output_direct, exist_ok=True)

    # Load pre-trained models
    model_ln = pickle.load(open(args.model_ln, "rb"))
    model_direct = pickle.load(open(args.model_direct, "rb"))

    # Traverse rivers
    for river_folder in sorted(os.listdir(args.input_dir)):
        subfolder_path = os.path.join(args.input_dir, river_folder)
        if not os.path.isdir(subfolder_path):
            continue

        tif_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(".tif")]

        for tif_file in tqdm(tif_files, desc=f"{river_folder}"):
            input_path = os.path.join(subfolder_path, tif_file)
            river_name = tif_file.split("_")[0]
            dem_path = get_dem_path(args.dem_dir, river_name)

            if not os.path.exists(dem_path):
                print(f"DEM not found: {dem_path}")
                continue

            with rasterio.open(dem_path) as dem_src:
                dem = dem_src.read(1).astype(np.float32)
                dem = np.where(np.isfinite(dem), dem, np.nan)
                dem_shape = dem_src.shape

            with rasterio.open(input_path) as src:
                if src.crs.to_string() != "EPSG:4326":
                    print(f"Invalid CRS (must be EPSG:4326): {tif_file}")
                    continue

                if src.shape != dem_shape:
                    print(f"Shape mismatch with DEM: {tif_file}")
                    continue

                scale_factor = 0.0000275
                blue = src.read(1).astype(np.float32)* scale_factor
                green = src.read(2).astype(np.float32)* scale_factor
                red = src.read(3).astype(np.float32)* scale_factor
                nir = src.read(4).astype(np.float32)* scale_factor
                swir1 = src.read(5).astype(np.float32)* scale_factor
                qa_pixel = np.nan_to_num(src.read(6), nan=0).astype(np.uint16)
                transform = src.transform
                height, width = src.shape
                profile = src.profile.copy()

            # Decode QA flags and generate valid mask
            qa = decode_qa_pixel(qa_pixel)
            qa_mask = (qa["dilated"] == 0) & (qa["cirrus"] == 0) & (qa["cloud"] == 0) & (qa["shadow"] == 0)

            # Spectral indices
            B3_B2 = np.divide(red, green)
            NDWI = (green - swir1) / (green + swir1)

            # Reflectance mask
            val_mask = (
                (green > 0) & (green <= 1) &
                (red > 0) & (red <= 1) &
                # (swir1 > 0) & (swir1 <= 0.1) &
                (NDWI >= 0)
            )
            final_mask = qa_mask & val_mask

            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            lon_flat = np.array(xs).flatten()
            lat_flat = np.array(ys).flatten()
            flat_mask = final_mask.flatten()

            transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
            lon_wgs, lat_wgs = transformer.transform(lon_flat, lat_flat)

            month = extract_month_from_filename(tif_file)
            if month is None:
                print(f"Month could not be parsed: {tif_file}")
                continue

            
            df = pd.DataFrame({
                'B3_B2': B3_B2.flatten()[flat_mask],
                'NDWI': NDWI.flatten()[flat_mask],
                'lat': lat_wgs[flat_mask],
                'lon': lon_wgs[flat_mask],
                'month': month,
                'SR_B3': red.flatten()[flat_mask],
                'SR_B4': nir.flatten()[flat_mask],
                'filled': dem.flatten()[flat_mask]
            })

            pred_ln = np.exp(model_ln.predict(df))
            pred_ssc = model_direct.predict(df)

            full_ln = np.full((height * width), np.nan, dtype=np.float32)
            full_ssc = np.full((height * width), np.nan, dtype=np.float32)
            full_ln[flat_mask] = pred_ln
            full_ssc[flat_mask] = pred_ssc

            ln_img = full_ln.reshape((height, width))
            ssc_img = full_ssc.reshape((height, width))

            profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

            out_ln = os.path.join(args.output_ln, tif_file.replace(".tif", "_exp.tif"))
            out_direct = os.path.join(args.output_direct, tif_file.replace(".tif", "_direct.tif"))

            with rasterio.open(out_ln, "w", **profile) as dst:
                dst.write(ln_img, 1)

            with rasterio.open(out_direct, "w", **profile) as dst:
                dst.write(ssc_img, 1)

# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    try:
        get_ipython
        args = argparse.Namespace(
            input_dir="/media/gultekin-erten/Yeni Birim/ssc/version_10/inputs",
            dem_dir="/media/gultekin-erten/Yeni Birim/ssc/version_10/DEMs",
            model_ln="/media/gultekin-erten/Yeni Birim/ssc/version_10/models/catboost_ln_ssc.pkl",
            model_direct="/media/gultekin-erten/Yeni Birim/ssc/version_10/models/catboost_direct_ssc.pkl",
            output_ln="/media/gultekin-erten/Yeni Birim/ssc/version_10/Outputs/Ln",
            output_direct="/media/gultekin-erten/Yeni Birim/ssc/version_10/Outputs/Direct"
        )
        main(args)
    except NameError:
        parser = argparse.ArgumentParser(description="Predict SSC using trained CatBoost models from Landsat imagery.")
        parser.add_argument("--input_dir", required=True, help="Directory with subfolders containing input .tif images")
        parser.add_argument("--dem_dir", required=True, help="Directory with DEM files named <River>_SRTM_DEM.tif")
        parser.add_argument("--model_ln", required=True, help="Path to CatBoost model for ln(SSC)")
        parser.add_argument("--model_direct", required=True, help="Path to CatBoost model for direct SSC")
        parser.add_argument("--output_ln", required=True, help="Output directory for exp(ln(SSC)) predictions")
        parser.add_argument("--output_direct", required=True, help="Output directory for direct SSC predictions")
        args = parser.parse_args()
        main(args)
