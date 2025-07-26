"""
Precipitation Data Merger for Monthly and Daily GPCP Products

This script extracts precipitation values for given coordinates and dates,
using both monthly and daily NetCDF data files. It supports different NetCDF formats
with varying coordinate variable names and handles longitude transformations (0–360 or -180–180).
Results are merged into a CSV file with precipitation statistics.

Usage:
-------
python precip_merge.py \
    --monthly_dir path/to/monthly_nc4 \
    --daily_dir path/to/daily_nc4 \
    --input_csv path/to/input.csv \
    --output_csv path/to/output.csv \
    [--windows 1 3 5 7 10 15 20 30]

Author: Gültekin Erten
License: MIT
"""

import os
import argparse
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ------------------------- Argument Parser -------------------------
parser = argparse.ArgumentParser(description="Extract and merge GPCP monthly and daily precipitation to CSV.")
parser.add_argument('--monthly_dir', type=str, required=True, help='Directory containing monthly .nc4 files')
parser.add_argument('--daily_dir', type=str, required=True, help='Directory containing daily .nc/.nc4 files')
parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV with "date", "lat", "lon" columns')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save output CSV with precipitation data')
parser.add_argument('--windows', type=int, nargs='+', default=[1, 3, 5, 7, 10, 15, 20, 30],
                    help='List of daily averaging windows in days (e.g. 1 3 5 7)')
args = parser.parse_args()

# ---------------------- Coordinate Name Helper ----------------------
def get_coord_names(ds):
    """Return correct names for lat and lon dimensions in the dataset"""
    lat_name = 'lat' if 'lat' in ds.variables else 'latitude'
    lon_name = 'lon' if 'lon' in ds.variables else 'longitude'
    return lat_name, lon_name

# ---------------------- Monthly File Info ----------------------
monthly_files = [f for f in os.listdir(args.monthly_dir) if f.endswith(".nc4")]
monthly_records = []
for fname in monthly_files:
    try:
        yyyymm = fname.split("_")[2]
        date = pd.to_datetime(yyyymm, format="%Y%m")
        monthly_records.append({"filename": fname, "month_year": date.to_period("M")})
    except:
        continue
df_monthly_files = pd.DataFrame(monthly_records)

# ---------------------- Load Input CSV ----------------------
df = pd.read_csv(args.input_csv)
df['date'] = pd.to_datetime(df['date'])
df['month_year'] = df['date'].dt.to_period('M')
df['month_year_prev'] = (df['date'] - pd.DateOffset(months=1)).dt.to_period('M')

# Merge filenames from monthly dataset
df = df.merge(df_monthly_files.rename(columns={"filename": "filename_t"}), on='month_year', how='left')
df = df.merge(df_monthly_files.rename(columns={"filename": "filename_t_1"}), left_on='month_year_prev', right_on='month_year', how='left', suffixes=('', '_drop'))
df = df.drop(columns=[col for col in df.columns if col.endswith('_drop')])

# ---------------------- Monthly Precip Extraction ----------------------
def get_monthly_precip(file_name, lat, lon):
    try:
        path = os.path.join(args.monthly_dir, file_name)
        ds = xr.open_dataset(path)
        lat_name, lon_name = get_coord_names(ds)
        lons = ds[lon_name].values
        lon_adj = lon % 360 if lons.max() > 180 else lon
        return ds['sat_gauge_precip'].sel({lat_name: lat, lon_name: lon_adj}, method='nearest').values.item()
    except Exception as e:
        print(f"[ERROR] Error fetching monthly precip data: {e}")
        return None

tqdm.pandas()
df['precip_t'] = df.progress_apply(lambda row: get_monthly_precip(row['filename_t'], row['lat'], row['lon']), axis=1)
df['precip_t_1'] = df.progress_apply(lambda row: get_monthly_precip(row['filename_t_1'], row['lat'], row['lon']), axis=1)
df['precip_2mo_avg'] = df[['precip_t', 'precip_t_1']].mean(axis=1)

# ---------------------- Daily File Mapper ----------------------
daily_files = [f for f in os.listdir(args.daily_dir) if f.endswith(".nc") or f.endswith(".nc4")]
file_date_map = {}

for f in daily_files:
    if f.startswith("GPCPDAY_L3"):
        try:
            date_str = f.split("_")[2]  # e.g., 20140226
            file_date_map[pd.to_datetime(date_str, format="%Y%m%d")] = f
        except:
            continue
    elif "_daily_d" in f:
        try:
            date_str = f.split("_daily_d")[1][:8]  # e.g., 19980210
            file_date_map[pd.to_datetime(date_str, format="%Y%m%d")] = f
        except:
            continue

# ---------------------- Daily Precip Rolling Average ----------------------
def get_daily_precip_avg(lat, lon, end_date, window_days):
    values = []
    for i in range(window_days):
        dt = end_date - pd.Timedelta(days=i)
        fname = file_date_map.get(dt, None)
        if fname:
            fpath = os.path.join(args.daily_dir, fname)
            try:
                ds = xr.open_dataset(fpath)
                lat_name, lon_name = get_coord_names(ds)
                lons = ds[lon_name].values
                lon_adj = lon % 360 if lons.max() > 180 else lon
                val = ds['precip'].sel({lat_name: lat, lon_name: lon_adj}, method='nearest').values.item()
                values.append(val)
            except Exception as e:
                print(f"[ERROR] Error fetching daily precip data: {e}")
                continue
    return sum(values) / len(values) if values else None

# Apply for all windows
for w in args.windows:
    col = f"precip_{w}d_avg"
    df[col] = df.progress_apply(lambda row: get_daily_precip_avg(row['lat'], row['lon'], row['date'], w), axis=1)

# ---------------------- Save Result ----------------------
df.to_csv(args.output_csv, index=False)
print(f"Finished: Precipitation statistics written to {args.output_csv}")
