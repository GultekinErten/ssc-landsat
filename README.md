#  Suspended Sediment Prediction Using Remote Sensing and Machine Learning

A Python-based framework to estimate Suspended Sediment Concentration (SSC) across the continental U.S. using 40 years of satellite imagery, hydrological, and topographic datasets. The workflow integrates USGS SSC/discharge data, Landsat reflectance, WhiteboxTools topographic analysis, GPCP precipitation, and multiple machine learning models.

---

## Project Features

- Automated USGS SSC + Discharge data collection  
- Terrain analysis from DEMs using WhiteboxTools  
- Precipitation feature extraction from satellite-based NetCDF  
- Raster feature sampling (e.g., slope, SPI, NDWI, valley depth)  
- ML model training (CatBoost, LightGBM, XGBoost, RF, Linear)  
- Supports both raw SSC and ln(SSC) targets  
- SHAP-based feature interpretation (coming soon)  

---

## Directory Structure

```bash
project/
├── data/                  # Input data (CSV, NetCDF, DEM, etc.)
├── outputs/               # Model outputs, graphs, and intermediate files
├── scripts/               # All processing & training scripts
├── models/                # Saved ML models (.pkl)
├── requirements.txt       # Dependencies
├── .gitignore
├── LICENSE.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

---

##  Core Scripts and Usage

### 1. Download SSC and Discharge Data

```bash
python scripts/download_usgs_ssc_discharge.py \
  --station 08313000 \
  --begin_date 1982-01-01 \
  --end_date 1993-07-12 \
  --output ./data/ssc_discharge.csv
```

> If `--station` is omitted, downloads all available USGS sites with SSC data.

---

### 2. DEM Terrain Analysis (WhiteboxTools)

```bash
python scripts/dem_topographic_analysis.py \
  --raster_names SRTM_NW SRTM_NE SRTM_SW SRTM_SE \
  --output_dir ./outputs/dem
```

---

### 3. Extract Topographic Features per Station

```bash
python scripts/extract_topo_features.py \
  --input_csv ./data/station_locations.csv \
  --raster_dir ./outputs/dem \
  --output_csv ./data/with_topo.csv
```

---

### 4. Merge GPCP Precipitation Data

```bash
python scripts/precip_merge.py \
  --monthly_dir ./data/gpcp/monthly \
  --daily_dir ./data/gpcp/daily \
  --input_csv ./data/with_topo.csv \
  --output_csv ./data/with_precip.csv \
  --windows 3 7 15
```

---

### 5. Train CatBoost Models (raw & ln)

```bash
python scripts/train_catboost.py \
  --input_csv ./data/with_precip.csv \
  --output_dir ./models
```

---

### 6. Compare ML Models (Optional)

```bash
python scripts/metrics.py \
  --data ./data/with_precip.csv \
  --output ./results/model_comparison.csv \
  --thresholds 1000
```

---


### 7. Download Monthly Landsat Mosaics

```bash
python scripts/landsat_downloader.py \
  --assets projects/ee-user/assets/Hudson projects/ee-user/assets/Columbia \
  --roi_names Hudson Columbia \
  --start_date 1990-01-01 \
  --end_date 2000-12-31 \
  --cloud_filter 30 \
  --output_dir ./outputs/landsat
```

> Generates and downloads monthly Landsat mosaics (L5, L7, L8, L9) for given regions using Earth Engine.


## Output Metrics

| Metric       | Description                          |
|--------------|--------------------------------------|
| `R2`         | Coefficient of determination          |
| `RMSE`       | Root Mean Square Error               |
| `MAE`        | Mean Absolute Error                  |
| `CVRMSE`     | Coeff. of Variation of RMSE          |
| `NSE`        | Nash–Sutcliffe Efficiency            |
| `KGE`        | Kling–Gupta Efficiency               |
| `MAPE`       | Mean Absolute Percentage Error       |
| `RRMSE`      | Relative RMSE                        |

---


### 8. Predict SSC from Landsat Inputs

```bash
python scripts/predict_model.py \
  --input_dir ./outputs/landsat \
  --dem_dir ./data/dem \
  --model_ln ./models/catboost_ln_ssc.pkl \
  --model_direct ./models/catboost_direct_ssc.pkl \
  --output_ln ./outputs/pred_ln \
  --output_direct ./outputs/pred_direct
```

> Predicts SSC using both ln(SSC) and direct models, masking non-water pixels based on NDWI. Reflectance values are scaled prior to prediction.

---

## Model Outputs

- **Direct predictions** are saved as `*_direct.tif`
- **Back-transformed ln(SSC)** predictions as `*_exp.tif`
- Predictions are masked using NDWI to exclude non-water pixels
- GeoTIFFs are in EPSG:4326 with `float32` data type and NaN as nodata

---

## License

This project is licensed under the [MIT License](LICENSE.txt).

---

## Author

**Gültekin Erten**  
Remote Sensing & Environmental Modeling  
MTA – General Directorate of Mineral Research and Exploration, Türkiye  
Contact: (add your email or GitHub Issues link)

---

## Future Work

- [ ] SHAP-based explainability visualization  
- [ ] Sentinel-2 support  
- [ ] Deployment via web interface  
- [ ] Transfer learning to other countries