"""
predict_model.py

This script generates SSC predictions using CatBoost models trained on Landsat surface reflectance bands
and topographic information (DEM). It supports both ln(SSC) and direct SSC model predictions and saves 
outputs as GeoTIFFs with water-only masking based on NDWI.

Author: Gültekin Erten
"""

import os
import numpy as np
import rasterio
from rasterio.transform import xy
import pickle
from catboost import CatBoostRegressor
from datetime import datetime
from tqdm import tqdm
import argparse

def get_dem_path(dem_dir, river_name):
    """Return the DEM file path for a given river name."""
    return os.path.join(dem_dir, f"{river_name}_SRTM_DEM.tif")

def extract_month_from_filename(filename):
    """Extract the month (1–12) from the Landsat filename."""
    try:
        parts = filename.replace(".tif", "").split("_")
        date_str = "_".join(parts[-3:])  # Example: '1984_06_01'
        return datetime.strptime(date_str, "%Y_%m_%d").month
    except:
        return None

def load_raster_as_array(path):
    """Load a single-band raster file and return the array and metadata."""
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile
        shape = src.shape
        crs = src.crs
    return array, profile, shape, crs

def main(args):
    """Main prediction loop for SSC modeling."""
    os.makedirs(args.output_ln, exist_ok=True)
    os.makedirs(args.output_direct, exist_ok=True)

    # Load models
    with open(args.model_ln, "rb") as f:
        model_ln = pickle.load(f)
    with open(args.model_direct, "rb") as f:
        model_direct = pickle.load(f)

    # Process all input .tif files
    tif_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".tif")]

    for tif_file in tqdm(tif_files, desc="Generating predictions"):
        tif_path = os.path.join(args.input_dir, tif_file)
        river = tif_file.split("_")[0]
        dem_path = get_dem_path(args.dem_dir, river)

        if not os.path.exists(dem_path):
            print(f"DEM not found: {dem_path}")
            continue

        # Load DEM
        dem, _, dem_shape, _ = load_raster_as_array(dem_path)

        # Load Landsat bands
        with rasterio.open(tif_path) as src:
            if src.crs.to_string() != "EPSG:4326":
                print(f"Invalid CRS (expected EPSG:4326): {tif_file}")
                continue

            if src.shape != dem_shape:
                print(f"Shape mismatch with DEM: {tif_file}")
                continue

            B1 = src.read(1).astype(np.float32)  # Blue
            B2 = src.read(2).astype(np.float32)  # Green
            B3 = src.read(3).astype(np.float32)  # Red
            B4 = src.read(4).astype(np.float32)  # NIR

            # Apply surface reflectance scale factor
            scale_factor = 0.0000275
            B1 *= scale_factor
            B2 *= scale_factor
            B3 *= scale_factor
            B4 *= scale_factor

            # Compute spectral indices
            B3_B2 = np.divide(B2, B1 + 1e-6)
            NDWI = (B2 - B4) / (B2 + B4 + 1e-6)
            water_mask = (NDWI > 0).astype(np.uint8)

            # Get month from filename
            month = extract_month_from_filename(tif_file)
            if month is None:
                print(f"Date parsing failed: {tif_file}")
                continue

            # Compute lat/lon for each pixel
            rows, cols = np.indices(B1.shape)
            lons, lats = xy(src.transform, rows, cols, offset='center')
            lons = np.array(lons, dtype=np.float32)
            lats = np.array(lats, dtype=np.float32)

            # Build feature matrix
            h, w = B1.shape
            features = np.stack([
                B3_B2.ravel(),
                dem.ravel(),
                lons.ravel(),
                B2.ravel(),
                B3.ravel(),
                lats.ravel(),
                np.full((h * w,), month)
            ], axis=1)

            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

            # ln(SSC) prediction and exponentiation
            preds_ln = model_ln.predict(features)
            exp_ssc = np.exp(preds_ln).reshape(h, w)
            masked_ln = np.where(water_mask == 1, exp_ssc, np.nan)
            ln_out_path = os.path.join(args.output_ln, tif_file.replace(".tif", "_exp.tif"))
            with rasterio.open(ln_out_path, "w", **profile) as dst:
                dst.write(masked_ln.astype(np.float32), 1)

            # Direct SSC prediction
            preds_direct = model_direct.predict(features)
            direct_ssc = preds_direct.reshape(h, w)
            masked_direct = np.where(water_mask == 1, direct_ssc, np.nan)
            direct_out_path = os.path.join(args.output_direct, tif_file.replace(".tif", "_direct.tif"))
            with rasterio.open(direct_out_path, "w", **profile) as dst:
                dst.write(masked_direct.astype(np.float32), 1)

if __name__ == "__main__":
    try:
        get_ipython
        # If running in an interactive environment (e.g., Spyder, Jupyter), use hardcoded arguments
        args = argparse.Namespace(
            input_dir="/media/gultekin-erten/Yeni Birim/ssc/version_10/inputs",
            dem_dir="/media/gultekin-erten/Yeni Birim/ssc/version_10/dems",
            model_ln="/media/gultekin-erten/Yeni Birim/ssc/version_10/models/catboost_ln_ssc.pkl",
            model_direct="/media/gultekin-erten/Yeni Birim/ssc/version_10/models/catboost_direct_ssc.pkl",
            output_ln="/media/gultekin-erten/Yeni Birim/ssc/version_10/Outputs/Ln",
            output_direct="/media/gultekin-erten/Yeni Birim/ssc/version_10/Outputs/Direct"
        )
        main(args)
    except NameError:
        # If running from command line, use argparse arguments
        parser = argparse.ArgumentParser(description="Predict SSC and ln(SSC) from Landsat imagery using CatBoost models.")
        parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .tif files.")
        parser.add_argument("--dem_dir", type=str, required=True, help="Directory containing DEM files named as <River>_SRTM_DEM.tif.")
        parser.add_argument("--model_ln", type=str, required=True, help="Path to trained CatBoost model for ln(SSC).")
        parser.add_argument("--model_direct", type=str, required=True, help="Path to trained CatBoost model for direct SSC.")
        parser.add_argument("--output_ln", type=str, required=True, help="Output directory for exp(ln(SSC)) predictions.")
        parser.add_argument("--output_direct", type=str, required=True, help="Output directory for direct SSC predictions.")
        args = parser.parse_args()
        main(args)
