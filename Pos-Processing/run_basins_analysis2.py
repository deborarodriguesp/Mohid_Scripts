import os
import sys
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from shapely.geometry import Point
from matplotlib.cm import ScalarMappable, viridis
from matplotlib.colors import LogNorm

# -------------------- CONFIGURAÇÕES --------------------
#Datasets
mohid = 1
era5 = 1  
observational = 1

#Dates
start_date = datetime(2011, 1, 1)
end_date   = datetime(2021, 1, 1)

# ---------------- Inputs  ----------------
root_dir     = r"F:\Debora\Tocantins\Results\HDF"

# ---------------- Bains ----------------
shapefile_dir = "F:/Debora/Water Balance/shapefiles/"
shapefiles = {
    "sono": os.path.join(shapefile_dir, "sono/sono.shp"),
    "parana": os.path.join(shapefile_dir, "parana/parana.shp"),
    "medio_tocantins": os.path.join(shapefile_dir, "medio_tocantins/medio_tocantins.shp"),
    "alto_tocantins": os.path.join(shapefile_dir, "alto_tocantins/alto_tocantins.shp"),
    "tocantins": os.path.join(shapefile_dir, "tocantins/tocantins.shp")
}

basins = {}
for name, shp in shapefiles.items():
    basins[name] = gpd.read_file(shp).to_crs(epsg=4326)

# ---------------- Estruturas ----------------
variables = ["RWC","WS","TS", "Infiltration"]
acc = {var: [] for var in variables}
spatial_avg = {var: [] for var in variables}

# ============================================================
# ======================= CONSTANTS =========================
# ============================================================
# Conversão m³/s → mm/h
dt = 3600  # segundos
DX_deg = 0.03
DY_deg = 0.04
lat = -9  # latitude média da região
DX_m = DX_deg * 111132
DY_m = DY_deg * (111320 * np.cos(np.deg2rad(lat)))
cell_area = DX_m * DY_m

# Image config
lon1, lon2, lat1, lat2 = -50.5, -45.5, -17, -4

mask_nan = None
reach_centers = None
latitude = None
longitude = None 

# ============================================================
# ========================== DEFS ============================
# ============================================================
def load_and_process_group(group, mask_nan=None, fill_invalid=None, rotate_flip=True):
    daily_data = []  
    for k in group.keys():
        data = group[k][:].astype(np.float32)
        if fill_invalid is not None:
            data [data <= fill_invalid] = np.nan
        # rotacionar / flip
        if rotate_flip:
            data = np.rot90(data, k=1)
            data = np.flipud(data)
        # aplicar máscara 2D
        if mask_nan is not None:
            data = np.where(mask_nan, np.nan, data)
        daily_data.append(data)
    return daily_data
# ============================================================
# ========================== MAIN ============================
# ============================================================

if mohid == 1:
    for date_folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, date_folder)

        if not os.path.isdir(folder_path):
            continue
        if not date_folder[:8].isdigit():
            continue
            
        year = int(date_folder[:4])
        month = int(date_folder[4:6])
        day = int(date_folder[6:8])
        ym = (year, month)
        file_date = datetime(year, month, day)    
        if not (start_date <= file_date <= end_date):
            continue
            
        print(f"{year}-{month:02d}-{day:02d}")    
               
        # ======================================================
        # ================== POROUS MEDIA (RWC) ================
        # ======================================================
        # parâmetro que temos a média!
        hdf_pm = os.path.join(folder_path, "PorousMedia_1.hdf5")
        if os.path.exists(hdf_pm):
            with h5py.File(hdf_pm, 'r') as f:
                    
                daily_data = []
                pm_group = f['Results/relative water content']  # acesso rápido
                keys = list(pm_group.keys())
                
                for k in keys:
                    data = pm_group[k][:]
                    data = np.where(data <= -99, np.nan, data)
                    data = np.nanmean(data, axis=0)
                    daily_data.append(data) 

                daily_data = np.rot90(daily_data, k=1, axes=(1,2))
                daily_data = np.flip(daily_data, axis=1)
                
                if mask_nan is None:
                    mask_nan = np.isnan(daily_data[0])

                if latitude is None:
                    latitude = f['Grid/Latitude'][:-1, :-1]
                    longitude = f['Grid/Longitude'][:-1, :-1]
                    if latitude.shape != mask_nan.shape:
                        latitude = latitude.T
                        longitude = longitude.T
                        
                daily_data = np.where(mask_nan, np.nan, daily_data)
                daily_data = np.nanmean(daily_data, axis=0)
                acc["RWC"].append(daily_data)
                
                daily_mean_spatial = np.nanmean(daily_data[~mask_nan])
                spatial_avg["RWC"].append(daily_mean_spatial)

        # ======================================================
        # ======================== STRESS ======================
        # ======================================================
        hdf_evtp = os.path.join(folder_path, "Vegetation_1.hdf5")
        if os.path.exists(hdf_evtp):
            with h5py.File(hdf_evtp, 'r') as f:
                daily_data = []
                veg_group = f['Results/WaterStress']
                daily_data = load_and_process_group(veg_group, mask_nan=mask_nan)
                daily_data = np.where(
                    np.all(np.isnan(daily_data), axis=0),
                    np.nan,
                    np.nanmean(daily_data, axis=0)
                )

                acc["WS"].append(daily_data)
                spatial_avg["WS"].append(np.nanmean(daily_data))
                
                daily_data = []
                veg_group = f['Results/TemperatureStress']
                daily_data = load_and_process_group(veg_group, mask_nan=mask_nan)
                daily_data = np.where(
                    np.all(np.isnan(daily_data), axis=0),
                    np.nan,
                    np.nanmean(daily_data, axis=0)
                )

                acc["TS"].append(daily_data)
                spatial_avg["TS"].append(np.nanmean(daily_data))
                
        # ======================================================
        # ======================== INFILT ======================
        # ======================================================
        hdf_basin = os.path.join(folder_path, "Basin_1.hdf5")
        if os.path.exists(hdf_evtp):
            with h5py.File(hdf_basin, 'r') as f:
                daily_data = []
                et_group = f['Results/AccInfiltration']
                daily_data = load_and_process_group(et_group, mask_nan=mask_nan)
                
                daily_total = daily_data[-1] - daily_data[0]
                daily_total_mm = daily_total * 1000
                daily_total_mm[np.isnan(daily_data[-1])] = np.nan
            
            acc["Infiltration"].append(daily_total_mm)
                
    print('✅ MOHID Land calculations done') 

# ======================================================
# ========= EXTRAÇÃO SÉRIE TEMPORAL POR BACIA ==========
# ======================================================
for k in acc:
    print(k, len(acc[k]))

dates_list = []
for i in range(len(acc["RWC"])):
    dates_list.append(start_date + timedelta(days=i))
    
accumulate_vars = ["Infiltration"] 
mean_vars = ["RWC","WS","TS"]
all_vars = accumulate_vars + mean_vars

# Flatten grid once
lat_flat = latitude.flatten()
lon_flat = longitude.flatten()
coords = np.column_stack((lon_flat, lat_flat))
num_cells = coords.shape[0]

grid_mask = {}  # máscara calculada uma única vez

if not grid_mask:
    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon_flat, lat_flat),
        crs="EPSG:4326"
    )

    for basin_name, basin_gdf in basins.items():
        points_in = gpd.sjoin(grid_points, basin_gdf, predicate="within")
        mask = np.zeros(latitude.shape, dtype=bool)

        for idx in points_in.index:
            i, j = np.unravel_index(idx, latitude.shape)
            mask[i, j] = True

        grid_mask[basin_name] = mask
        
results = {name: [] for name in basins.keys()}

for basin_name, mask in grid_mask.items():
    # Usamos uma variável de referência para o tempo (ex: RWC ou PM)
    for t in range(len(acc[next(iter(acc))])): 
        rec = {"Date": dates_list[t], "Basin": basin_name}
        skip = False

        for var_name in all_vars:
            # 1. Segurança: Verifica se a variável existe e é válida
            if not var_name or var_name not in acc:
                continue
            
            data_t = acc[var_name][t]
            masked = np.where(mask, data_t, np.nan)
            
            # 2. Verifica se a bacia está vazia de dados (apenas NaNs)
            if np.all(np.isnan(masked)):
                value = np.nan
            else:
                # 3. Cálculo da Mediana (mais robusto contra extremos)
                value = np.nanmean(masked)
            
            rec[var_name] = value
            
            # 4. Lógica de descarte: 
            # SÓ pulamos se for NaN (falta de imagem/dado). 
            # Se for 0.0, mantemos no resultado!
            if np.isnan(value):
                skip = True

        if not skip:
            results[basin_name].append(rec)

# Criar DataFrame
df_daily_basin = pd.DataFrame([r for recs in results.values() for r in recs])

output_dir = "basin_outputs2"
os.makedirs(output_dir, exist_ok=True)

for basin_name in basins.keys():
    
    # ---------------- DAILY ----------------
    df_daily = df_daily_basin[df_daily_basin["Basin"] == basin_name]
    daily_path = os.path.join(output_dir, f"{basin_name}_daily.csv")
    df_daily.to_csv(daily_path, index=False)
    print(f"✅ Daily exported: {daily_path}")
   