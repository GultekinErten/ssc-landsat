"""
Monthly Landsat Mosaic Downloader via Google Earth Engine

This script downloads monthly mosaic images for one or more regions using Landsat 5, 7, 8, and 9 imagery from
Google Earth Engine. It filters images by cloud cover, masks snow and cloud/shadow/cirrus pixels using QA_PIXEL,
and exports clipped .tif images per region and date.

Usage:
-------
python landsat_downloader.py \
    --assets projects/ee-user/assets/Hudson projects/ee-user/assets/Columbia \
    --roi_names Hudson Columbia \
    --start_date 2000-01-01 \
    --end_date 2005-12-31 \
    --cloud_filter 20 \
    --output_dir ./outputs/landsat

Notes:
- Requires Earth Engine Python API and geemap package.
- Each output TIFF is named as <ROI>_<Satellite>_<YYYY_MM_DD>.tif
- Filters L7 by default to avoid SLC-off period (post-2003-05-30).

Author: Gültekin Erten  
License: MIT
"""

import ee
import geemap
import os
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
import sys
sys.argv = [
    'landsat_downloader.py',
    '--assets',
    'projects/ee-gultekinerten/assets/Colorado',
    '--roi_names',
    'Colorado',
    '--start_date', '1984-01-01',
    '--end_date', '2025-12-31',
    '--cloud_filter', '100',
    '--output_dir', '/media/gultekin-erten/Yeni Birim/ssc/version_10/inputs'
]

# ----------------------------- Argument Parser -----------------------------
parser = argparse.ArgumentParser(
    description="Download monthly mosaics from Landsat collections using Google Earth Engine."
)
parser.add_argument("--assets", nargs='+', required=True,
                    help="List of Earth Engine asset paths for ROIs (e.g., 'projects/.../Hudson')")
parser.add_argument("--roi_names", nargs='+', required=True,
                    help="List of region names corresponding to the assets (e.g., 'Hudson Columbia')")
parser.add_argument("--start_date", type=str, default="1984-01-01",
                    help="Start date in format YYYY-MM-DD (default: 1984-01-01)")
parser.add_argument("--end_date", type=str, default="2024-12-31",
                    help="End date in format YYYY-MM-DD (default: 2024-12-31)")
parser.add_argument("--cloud_filter", type=int, default=50,
                    help="Maximum cloud cover percentage for filtering (default: 50)")
parser.add_argument("--output_dir", type=str, default="~/Desktop/landsat_downloads",
                    help="Directory to save downloaded TIFFs (default: ~/Desktop/landsat_downloads)")
args = parser.parse_args()

# ------------------------------ Earth Engine Init ------------------------------
ee.Initialize()

output_dir = os.path.expanduser(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

if len(args.assets) != len(args.roi_names):
    raise ValueError("Length of --assets and --roi_names must match.")

# ------------------------ Landsat Collections & Bands ------------------------
landsat_collections = {
    "L5": "LANDSAT/LT05/C02/T1_L2",
    "L7": "LANDSAT/LE07/C02/T1_L2",
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2"
}

bands_dict = {
    "L5": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
    "L7": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
    "L8": ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'QA_PIXEL'],
    "L9": ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'QA_PIXEL']
}

# ------------------------------ Helper Functions ------------------------------
def filter_clear_images(image):
    """Add 'bad_pct' property to each image based on cloud, shadow, and cirrus pixels."""
    qa = image.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    cirrus = qa.bitwiseAnd(1 << 2).neq(0)
    any_bad = cloud.Or(shadow).Or(cirrus)

    bad_pct = any_bad.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('QA_PIXEL')

    return image.set('bad_pct', bad_pct)

def mask_snow(image):
    """Mask out snow pixels using QA_PIXEL band."""
    qa = image.select('QA_PIXEL')
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    return image.updateMask(snow)

def generate_monthly_mosaics(collection, start_date, end_date):
    """Generate a list of (mosaic_image, date_str) tuples for each month."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    mosaics = []

    while start <= end:
        month_start = ee.Date(start.strftime("%Y-%m-%d"))
        month_end = month_start.advance(1, 'month')
        filtered = collection.filterDate(month_start, month_end)

        try:
            count = filtered.size().getInfo()
        except Exception as e:
            print(f"[Warning] Could not get image count for {start.strftime('%Y-%m')}: {e}")
            count = 0

        if count > 0:
            try:
                mosaic = filtered.mosaic()
                date_str = start.strftime("%Y_%m_%d")
                mosaics.append((mosaic, date_str))
                print(f"Mosaic ready for {date_str}")
            except Exception as e:
                print(f"[Error] Mosaic creation failed for {start.strftime('%Y-%m')}: {e}")

        start += timedelta(days=32)
        start = datetime(start.year, start.month, 1)

    print(f"{len(mosaics)} mosaics generated")
    return mosaics

# ------------------------------ Main Loop ------------------------------
for roi_name, roi_path in zip(args.roi_names, args.assets):
    roi = ee.FeatureCollection(roi_path)
    region = roi.geometry().bounds()
    print(f"\nRegion: {roi_name}")

    for tag, collection_id in landsat_collections.items():
        print(f"  Processing Landsat collection: {tag}")

        col = ee.ImageCollection(collection_id) \
            .filterDate(args.start_date, args.end_date) \
            .filterBounds(roi) \
            .filter(ee.Filter.lte('CLOUD_COVER', args.cloud_filter))

        if tag == "L7":
            col = col.filterDate('1999-01-01', '2003-05-30')

        col = col.map(filter_clear_images)
        col = col.filter(ee.Filter.Or(
            ee.Filter.notNull(['bad_pct']),
            ee.Filter.lte('bad_pct', 0.0001)
        ))
        col = col.map(mask_snow)

        mosaics = generate_monthly_mosaics(col, args.start_date, args.end_date)

        for i, (image, date_str) in enumerate(tqdm(mosaics, desc=f"{roi_name} - {tag}")):
            filename = f"{roi_name}_{tag}_{date_str}.tif"
            tif_path = os.path.join(output_dir, filename)

            try:
                print(f"Downloading {i+1}/{len(mosaics)}: {filename}")
                image_selected = image.select(bands_dict[tag]).clip(region).unmask()

                geemap.download_ee_image(
                    image=image_selected,
                    filename=tif_path,
                    region=region,
                    scale=30,
                    crs='EPSG:4326'
                )

                print(f"Saved: {filename}")

            except Exception as e:
                print(f"[Error] Failed to download {filename}: {e}")