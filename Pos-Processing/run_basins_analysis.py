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
era5_folder  = r"E:\Debora\Interpolated_Watershed"
precip_folder = r"F:\Debora\Araguaia"
hdf_path      = os.path.join(precip_folder, "chuva_observada.hdf5")   
 
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

# ---------------- Drainage Network ----------------
dn_file = r"F:\Debora\Water Balance\DrainageNetwork\tocantins.shp"
dn_gdf = gpd.read_file(dn_file)

# garantir CRS (assumindo lon/lat)
if dn_gdf.crs is None:
    dn_gdf = dn_gdf.set_crs(epsg=4326)
else:
    dn_gdf = dn_gdf.to_crs(epsg=4326)

# ---------------- Estruturas ----------------
variables = ["FLOW","PM","ET","RO","RO_FLOW","Pobs","P","T"]
acc = {var: [] for var in variables}
spatial_avg = {var: [] for var in variables if var != "FLOW"}

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
    daily_list = []  
    for k in group.keys():
        data = group[k][:]
        if fill_invalid is not None:
            data = np.where(data <= fill_invalid, np.nan, data)
        daily_list.append(data)
    daily_data = np.stack(daily_list)
    
    if rotate_flip:
        daily_data = np.rot90(daily_data, k=1, axes=(1, 2))
        daily_data = np.flip(daily_data, axis=1)
        
    if mask_nan is not None:
        daily_data = np.where(mask_nan, np.nan, daily_data)
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
        # ================== POROUS MEDIA (WTD) ================
        # ======================================================
        # parâmetro que temos a média!
        hdf_pm = os.path.join(folder_path, "PorousMedia_1.hdf5")
        if os.path.exists(hdf_pm):
            with h5py.File(hdf_pm, 'r') as f:
                    
                daily_data = []
                pm_group = f['Results/water table depth']  # acesso rápido
                keys = list(pm_group.keys())
                
                for k in keys:
                    data = pm_group[k][:]
                    data = np.where(data <= -99, np.nan, data)
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
                acc["PM"].append(daily_data)
                
                daily_mean_spatial = np.nanmean(daily_data[~mask_nan])
                spatial_avg["PM"].append(daily_mean_spatial)           
                    
        # ======================================================
        # ========================= EVTP =======================
        # ======================================================
        hdf_evtp = os.path.join(folder_path, "EVTP_1.hdf5")
        if os.path.exists(hdf_evtp):
            with h5py.File(hdf_evtp, 'r') as f:
                daily_data = []
                et_group = f['Results/AccEVTP']
                daily_data = load_and_process_group(et_group, mask_nan=mask_nan)
                daily_sum = np.where(
                    np.all(np.isnan(daily_data), axis=0),
                    np.nan,
                    np.nansum(daily_data, axis=0)
                )

                acc["ET"].append(daily_sum)
                spatial_avg["ET"].append(np.nanmean(daily_sum))

        # ======================================================
        # ======================== RUNOFF ======================
        # ======================================================
        # TRANSFORMAR PARA MM #
        hdf_ro = os.path.join(folder_path, "Runoff_1.hdf5")
        if os.path.exists(hdf_ro):
            with h5py.File(hdf_ro, 'r') as f:
                ro_group = f['Results/flow modulus']
                daily_data = load_and_process_group(ro_group, mask_nan=mask_nan)
                
                # -----------Acúmulo diário em mm (para recharge) -------------
                daily_sum = np.where(
                    np.all(np.isnan(daily_data), axis=0),
                    np.nan,
                    np.nansum(daily_data *dt, axis=0)
                )    
                # ----------- Conversão para mm -------------
                # conversão para mm, preservando NaN
                runoff_mm = np.where(
                    np.isnan(daily_sum),
                    np.nan,
                    (daily_sum / cell_area) * 1000.0
                )

                # ----------- MÉDIA DIÁRIA EM m³/s -------------
                daily_mean = np.where(
                    np.all(np.isnan(daily_data), axis=0),
                    np.nan,
                    np.nanmean(daily_data, axis=0)  # m³/s
                )
                
                acc["RO"].append(runoff_mm)
                acc["RO_FLOW"].append(daily_mean)

                # ---------- Média espacial diária ----------
                spatial_avg["RO"].append(np.nanmean(runoff_mm[~mask_nan]))
                spatial_avg["RO_FLOW"].append(np.nanmean(daily_mean[~mask_nan]))
   
    print('✅ MOHID Land calculations done') 

# ======================================================
# ========================= ERA5 =======================
# ======================================================
if era5 == 1:
    for filename in sorted(os.listdir(era5_folder)):
        if not filename.endswith(".hdf5"):
            print('Files not found')
            continue

        file_date = datetime.strptime(filename[:8], "%Y%m%d")
        if not (start_date <= file_date <=  end_date):
            continue  # ignorar arquivos fora do intervalo

        era_path = os.path.join(era5_folder, filename)
        print("ERA5: ",file_date)
        
        with h5py.File(era_path, 'r') as f:
            ym = (file_date.year, file_date.month)
            
            # ---------- PRECIPITATION ----------
            if "Results/precipitation" not in f:
                print(f"⚠️ Variável 'precipitation' não encontrada em {filename}. Pulando arquivo.")
                continue
                
            daily_data = []
            p_group = f['Results/precipitation']
            
            daily_data = load_and_process_group(p_group, mask_nan=mask_nan) 
            daily_data = np.where(
                np.all(np.isnan(daily_data), axis=0),
                np.nan,
                np.nansum(daily_data, axis=0)
            ) 
             
            acc["P"].append(daily_data)
            spatial_avg["P"].append(np.nanmean(daily_data))
            
            # ---------- AIR TEMPERATURE ----------    
            daily_data = []
            t_group = f['Results/air temperature']  # acesso rápido
            daily_data = load_and_process_group(t_group, mask_nan=mask_nan)                
            daily_data = np.where(
                np.all(np.isnan(daily_data), axis=0),
                np.nan,
                np.nanmean(daily_data, axis=0)
            )  
            acc["T"].append(daily_data)
            spatial_avg["T"].append(np.nanmean(daily_data))
            
    print("✅ ERA5 precipitation and temperature done")
        
# ======================================================
# ========================= Observational ==============
# ======================================================
if observational == 1:
    with h5py.File(hdf_path, 'r') as f:
        p_group = f['Results/precipitation']
        p_keys = list(p_group.keys())

        t_group = f['Time']  # (ntimes, 6)
        t_keys = list(t_group.keys())
        
        for tk, pk in zip(t_keys, p_keys):
            # ---------- TEMPO ----------
            t = t_group[tk][:]
            file_date = datetime(int(t[0]), int(t[1]), int(t[2]))
            if not (start_date <= file_date <= end_date):
                continue
                
            print("OBS: ", file_date)
            
            # ---------- PRECIPITAÇÃO DIÁRIA ----------
            data = p_group[pk][:]  # (ny, nx)
            data = np.where(data < 0, np.nan, data)
            data = np.rot90(data, k=1)
            data = np.flipud(data)
            data = np.where(mask_nan, np.nan, data)
            
            acc["Pobs"].append(data)
            spatial_avg["Pobs"].append(np.nanmean(data))
            
    print('✅ Observational Precipitation done')

# ======================================================
# ========= EXTRAÇÃO SÉRIE TEMPORAL POR BACIA ==========
# ======================================================
for k in acc:
    print(k, len(acc[k]))

dates_list = []
for i in range(len(acc["PM"])):
    dates_list.append(start_date + timedelta(days=i))
    
accumulate_vars = ["P", "Pobs", "ET", "RO"] 
mean_vars = ["PM","RO_FLOW", "T"]
all_vars = accumulate_vars + mean_vars

# Flatten grid once
lat_flat = latitude.flatten()
lon_flat = longitude.flatten()
coords = np.column_stack((lon_flat, lat_flat))
num_cells = coords.shape[0]

grid_mask = {}  # máscara calculada uma única vez
results = {name: [] for name in basins.keys()}
basin_stats = {}

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

# ======================================================
# ========= CRIAÇÃO DA MÁSCARA DE BACIAS ===============
# ======================================================
for basin_name, basin_gdf in basins.items():
    points_in = gpd.sjoin(grid_points, basin_gdf, predicate="within")
    mask = np.zeros(latitude.shape, dtype=bool)

    for idx in points_in.index:
        i, j = np.unravel_index(idx, latitude.shape)
        mask[i, j] = True
    grid_mask[basin_name] = mask

for basin_name, mask in grid_mask.items():
    
    total_cells_mask = np.sum(mask)
    et_active_counts = []
    
    for t in range(len(acc["PM"])):
        rec = {"Date": dates_list[t], "Basin": basin_name}
        skip = False
        
        for var_name in all_vars:
            data_t = acc[var_name][t]
            masked = np.where(mask, data_t, np.nan)
            
            # --- NOVO FILTRO DE OUTLIER ---
            if var_name == "ET":
                # Filtro: só o que é maior que 0
                et_filter = (masked > 0.00001)
                cells_t = np.sum(et_filter)
                et_active_counts.append(cells_t) # Salva para a média final
                value = np.nanmean(np.where(et_filter, masked, np.nan))
            else:
                value = np.nanmean(masked)
            
            rec[var_name] = value
            if np.isnan(value): # Se a bacia toda falhar
                skip = True
                break
                
        if not skip:
            results[basin_name].append(rec)
            
    # --- CÁLCULO DA MÉDIA DE ÁREA DA BACIA ---
    avg_active_cells = np.mean(et_active_counts) if et_active_counts else 0
    pct_utilizada = (avg_active_cells / total_cells_mask) * 100 if total_cells_mask > 0 else 0
    
    basin_stats[basin_name] = {
        "Total_Cells": total_cells_mask,
        "Avg_Active_ET_Cells": round(avg_active_cells, 2),
        "Pct_Area_Media_Utilizada": round(pct_utilizada, 2)
    }

# Exibe o resumo no console
print("\n=== RESUMO DE ÁREA ÚTIL (MÉDIA DO PERÍODO) ===")
for b, stats in basin_stats.items():
    print(f"Bacia: {b} | Células Totais: {stats['Total_Cells']} | "
          f"Uso Médio ET: {stats['Pct_Area_Media_Utilizada']}%")
          
          
# Criar DataFrame
df_daily_basin = pd.DataFrame([r for recs in results.values() for r in recs])

output_dir = "basin_outputs"
os.makedirs(output_dir, exist_ok=True)

if df_daily_basin.empty:
    print("❌ ERRO: O DataFrame está vazio! Verifique se os dados de entrada (acc) contêm valores válidos.")
else:
    for basin_name in basins.keys():
        df_daily = df_daily_basin[df_daily_basin["Basin"] == basin_name]
        daily_path = os.path.join(output_dir, f"{basin_name}_daily.csv")
        df_daily.to_csv(daily_path, index=False)
        print(f"✅ Daily exported: {daily_path}")


import seaborn as sns

def plot_global_histograms(acc, grid_mask, basin_name, output_dir):
    """
    Gera uma matriz de histogramas para todas as variáveis da bacia
    considerando todo o tempo disponível.
    """
    vars_to_plot = [v for v in acc.keys() if v in ["P", "ET", "T", "PM"]]
    num_vars = len(vars_to_plot)
    
    # Criar subplots dinamicamente
    fig, axes = plt.subplots(nrows=(num_vars + 1) // 2, ncols=2, figsize=(12, num_vars * 2))
    axes = axes.flatten()
    
    mask = grid_mask[basin_name]

    for i, var_name in enumerate(vars_to_plot):
        # Acumular todos os pixels de todos os dias para esta bacia
        all_values = []
        for t in range(len(acc[var_name])):
            data_t = acc[var_name][t]
            # Pega apenas os pixels da bacia e remove NaNs
            masked_vals = data_t[mask]
            # --- FILTRO ESPECÍFICO PARA ET ---
            if var_name == "ET":
                valid_vals = valid_vals[valid_vals > 0.00001]
            all_values.append(valid_vals)
        
        # Transforma em um único array longo
        final_array = np.concatenate(all_values)
        
        # Plotar no subplot correspondente
        sns.histplot(final_array, bins=50, ax=axes[i], kde=True, color='teal')
        
        axes[i].set_title(f'Distribuição Global: {var_name} ({basin_name})')
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')
        
        # Adicionar info de média e mediana no gráfico
        avg = np.mean(final_array)
        med = np.median(final_array)
        axes[i].axvline(avg, color='red', linestyle='--', label=f'Média: {avg:.2f}')
        axes[i].axvline(med, color='yellow', linestyle='-', label=f'Mediana: {med:.2f}')
        axes[i].legend()

    # Ajustar layout e remover eixos extras se houver
    plt.tight_layout()
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    hist_path = os.path.join(output_dir, f"histograms_global_{basin_name}.png")
    plt.savefig(hist_path, dpi=300)
    print(f"✅ Histograma global salvo: {hist_path}")
    plt.close()

# Chamada da função (coloque após o loop de extração das máscaras)
for basin in basins.keys():
    plot_global_histograms(acc, grid_mask, basin, output_dir)