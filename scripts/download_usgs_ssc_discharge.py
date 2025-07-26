"""
USGS NWIS SSC and Discharge Data Downloader

This script downloads Suspended Sediment Concentration (SSC) and Discharge (Q)
data for a given time range from the USGS NWIS website. It supports scraping
station metadata, converting coordinates from NAD27/NAD83 to WGS84, and merging
SSC and Discharge data into a single CSV file.

Usage:
------
python usgs_nwis_downloader.py \
    --begin_date 2022-01-01 \
    --end_date 2023-12-31 \
    --output data/output.csv

Optional:
    --station 08158000              # One station or comma-separated list

Author: Gültekin Erten
License: MIT
"""

import os
import requests
from bs4 import BeautifulSoup
from pyproj import Transformer
import pandas as pd
import argparse
from tqdm import tqdm
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Disable SSL warnings ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Retry session ---
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retry))
session.mount("https://", HTTPAdapter(max_retries=retry))

def safe_get(url, **kwargs):
    try:
        return session.get(url, timeout=10, verify=False, **kwargs)
    except Exception as e:
        print(f"[ERROR] Skipping {url}: {e}")
        return None

# --- Coordinate Conversion ---
def nad27_to_wgs84(lon, lat):
    transformer = Transformer.from_crs("epsg:4267", "epsg:4326", always_xy=True)
    return transformer.transform(lon, lat)

def nad83_to_wgs84(lon, lat):
    transformer = Transformer.from_crs("epsg:4269", "epsg:4326", always_xy=True)
    return transformer.transform(lon, lat)

def dms_to_dd(d, m, s):
    return float(d) + float(m) / 60 + float(s) / 3600

# --- Argparse Setup ---
parser = argparse.ArgumentParser(description="Download SSC and Discharge data from USGS NWIS")
parser.add_argument("--station", type=str, default=None, help="Comma-separated USGS Station IDs")
parser.add_argument("--begin_date", type=str, required=True, help="Start date in YYYY-MM-DD")
parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD")
parser.add_argument("--output", type=str, required=True, help="Output CSV path")
args = parser.parse_args()

# --- Prepare output path ---
if not args.output.endswith(".csv"):
    args.output += ".csv"
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# --- Station List ---
if args.station:
    station_list = args.station.split(',')
else:
    url = "https://waterdata.usgs.gov/nwis/dv?..."  # trimmed for brevity
    resp = safe_get(url)
    if resp is None:
        raise RuntimeError("Failed to retrieve station list.")
    text = resp.text.split("PROVISIONAL DATA SUBJECT TO REVISION")[1].split("site_no")
    text.pop(0)
    station_list = [text[i].split("=")[1].split('"')[0] for i in range(len(text)-2)]

# --- Containers ---
ssc_records = []
discharge_records = []

# --- Download Loop ---
def get_latlon(site_no):
    resp = safe_get(f"https://waterdata.usgs.gov/nwis/inventory/?site_no={site_no}&agency_cd=USGS")
    if not resp: return None, None
    try:
        soup = BeautifulSoup(resp.content, "html.parser")
        dms = soup.find_all("dd")[0].get_text(strip=True).replace('\xa0', ' ').split(",")
        # Parse latitude
        lat_d, lat_m, lat_s = map(float, dms[0].split("Latitude ")[1].split()[1].replace("\u00b0", " ").replace("'", " ").replace('"', '').split())
        lat_dd = dms_to_dd(lat_d, lat_m, lat_s)
        # Parse longitude
        lon_parts = dms[1].split("Longitude ")[1].strip().split()
        lon_d, lon_m, lon_s = map(float, lon_parts[0].replace("\u00b0", " ").replace("'", " ").replace('"', '').split())
        lon_dd = -dms_to_dd(lon_d, lon_m, lon_s)
        datum = lon_parts[1]
        if datum == "NAD27":
            return nad27_to_wgs84(lon_dd, lat_dd)
        elif datum == "NAD83":
            return nad83_to_wgs84(lon_dd, lat_dd)
        else:
            return lon_dd, lat_dd
    except:
        return None, None

def download_data(site_no, param_code, label):
    data = []
    url = f"https://waterdata.usgs.gov/nwis/dv?cb_{param_code}=on&format=html&site_no={site_no}&begin_date={args.begin_date}&end_date={args.end_date}"
    resp = safe_get(url)
    if not resp or "There are no data available" in resp.text:
        return data
    try:
        parts = resp.text.split("tbody")[3].split("nowrap")
        lon, lat = get_latlon(site_no)
        for i in range(1, len(parts)//4):
            value = parts[i*4].split("span")[1][:-2][1:]
            value = value.replace(",", "")
            value = "0" if value == '&nbsp;' else value
            try:
                value = int(float(value))
            except:
                value = None
            date = f"{parts[i*4-2][9:13]}-{parts[i*4-2][3:5]}-{parts[i*4-2][6:8]}"
            data.append((site_no, date, lat, lon, value))
    except:
        pass
    return data

# --- Main Download ---
for site in tqdm(station_list, desc="Downloading SSC"):
    ssc_records.extend(download_data(site, "80154", "SSC"))
for site in tqdm(station_list, desc="Downloading Discharge"):
    discharge_records.extend(download_data(site, "00060", "Discharge"))

# --- Save CSV ---
df_ssc = pd.DataFrame(ssc_records, columns=["USGS_ID", "Date", "Latitude", "Longitude", "SSC"])
df_q = pd.DataFrame(discharge_records, columns=["USGS_ID", "Date", "Latitude", "Longitude", "Discharge"])
df_final = pd.merge(df_ssc, df_q, on=["USGS_ID", "Date", "Latitude", "Longitude"], how="outer")
df_final.to_csv(args.output, index=False)
print(f"Saved merged SSC and Discharge data to: {args.output}")