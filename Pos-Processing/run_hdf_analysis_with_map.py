import os, sys,  h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, viridis
from matplotlib.colors import LogNorm, Normalize
from collections import defaultdict
matplotlib.use('Agg')
import gc

import contextily as ctx

# -------------------- CONFIGURAÇÕES --------------------
#Datasets
mohid = 1      
era5 = 1       
observational = 0
plot = 1

#Dates
start_date = datetime(2025, 1, 1)
end_date   = datetime(2026, 3, 10)

# ---------------- Inputs  ----------------
# Dir MOHID
root_dir = r"D:\DOUTORAMENTO\TwinStream\Projetos\TwinStream_ModelosHidrologicos\Modelos\Mondego_Corrigido_reservoir\Results\HDF"
era5_folder = r"\\ml5-irrigserv\TwinStream\meteomondego"
precip_folder = r"E:\Modelos\Observational_precipitation\TAW_glue"
hdf_path = os.path.join(precip_folder, "chuva_observada.hdf5")

# ---------------- Bains ----------------
shapefile_dir = r"D:\DOUTORAMENTO\TwinStream\Projetos\TwinStream_ModelosHidrologicos\Mapas\Portugal_shapefile\Extracted\basins_dissolved\Main"
shapefiles = {
    #"algarve": os.path.join(shapefile_dir, "algarve.shp"),
    #"cantabrian": os.path.join(shapefile_dir, "cantabrian_final.shp"),
    #"cavado_ave_leca": os.path.join(shapefile_dir, "cavado_ave_leca.shp"),
    #"douro": os.path.join(shapefile_dir, "Douro.shp"),
    #"galicia": os.path.join(shapefile_dir, "galicia.shp"),
    #"guadiana": os.path.join(shapefile_dir, "guadiana_final.shp"),
    #"minho": os.path.join(shapefile_dir, "minho_final.shp"),
    "mondego": os.path.join(shapefile_dir, "mondego.shp"),
    #"sado": os.path.join(shapefile_dir, "sado_final3.shp"),
    #"tejo": os.path.join(shapefile_dir, "tejo.shp")
}
print (shapefiles)

basins_gdf = []
for shp_path in shapefiles.values():
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    basins_gdf.append(gdf)

basins_gdf = gpd.GeoDataFrame(
    pd.concat(basins_gdf, ignore_index=True),
    crs="EPSG:4326"
)

# ---------------- Drainage Network ----------------
#dn_file = r"D:\DOUTORAMENTO\TwinStream\Projetos\TwinStream_ModelosHidrologicos\Inputs\Rivers\Monitorizacao\DN\DrainageNetwork_Douro.shp"
#dn_gdf = gpd.read_file(dn_file)
#
## garantir CRS (assumindo lon/lat)
#if dn_gdf.crs is None:
#    dn_gdf = dn_gdf.set_crs(epsg=4326)
#else:
#    dn_gdf = dn_gdf.to_crs(epsg=4326)
#
# ---------------- Estruturas ----------------
variables = ["FLOW","PM","ET","RO","RO_FLOW", "Infiltration","Pobs","P","T"] # SR
acc = {var: [] for var in variables}
spatial_avg = {var: [] for var in variables if var != "FLOW"} 
  
# ============================================================
# ======================= CONSTANTS =========================
# ============================================================
# Conversão m³/s → mm/h
dt = 3600  # segundos
DX_deg = 0.02
DY_deg = 0.02
lat = 40  # latitude média da região

DX_m = DX_deg * 111132
DY_m = DY_deg * (111320 * np.cos(np.deg2rad(lat)))
cell_area = DX_m * DY_m

#image configuration
lat1= 39.6 #south 
lat2= 40.9 #north
lon1= -9 #west
lon2= -7.15 #east

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
        
def compute_water_balance(P_dict, ET_dict, R_dict=None):
    recharge, deficiency, ratio_etp = {}, {}, {}
    for k in P_dict.keys():
        P  = P_dict[k]
        ET = ET_dict[k]
        R  = R_dict[k] if R_dict is not None else np.zeros_like(P)
        deficiency[k] = P - ET
        recharge[k] = np.maximum(P - ET - R, 0)
        P_safe = np.where(P<=1, 1, P)
        ratio_etp[k] = np.divide(ET, P_safe, out=np.full_like(ET,np.nan), where=~np.isnan(P_safe))
    return recharge, deficiency, ratio_etp

def plot_images(data_dict, title_prefix, cbar_label, filename_prefix, output_dir, cmap='viridis',
                vmin=None, vmax=None, is_grid=True, is_flow=False, log_scale=False, period='monthly', 
                zero_gray=False, drainage_gdf=None, basins_gdf=None):  

    # ---------- escala global ----------
    all_vals = []
    for v in data_dict.values():
        if is_grid:
            all_vals.append(v.flatten())
        else:
            all_vals.append(v)

    all_vals = np.concatenate(all_vals)
    all_vals = all_vals[~np.isnan(all_vals)]

    flow_min = 1e-1
    if log_scale:
        all_vals = all_vals[all_vals > flow_min]  # remove zeros para LogNorm
                
    if vmax is None:
        vmax_raw = np.nanpercentile(all_vals, 99) if len(all_vals)>0 else 1
        #vmax = np.nanmax(all_vals)
        if vmax_raw <= 3:
            # Arredonda para a próxima casa decimal (ex: 0.73 -> 0.8 ou 0.75)
            vmax = np.ceil(vmax_raw * 10) / 10
        else:
            vmax = np.ceil(vmax_raw / 5) * 5
            vmax = np.round(vmax).astype(int)
        vmax = float(vmax)
        print('vmax',vmax)
        
    if vmin is None:
        # Pega o menor valor maior que zero
        vmin_raw = np.nanmin(all_vals[all_vals > 0]) if len(all_vals) > 0 else 0.1
        
        if log_scale:
            # Para LOG, o vmin DEVE ser positivo. 
            # Se o vmin_raw for muito baixo, 0.1 é um bom limite inferior para mapas de mm.
            vmin = max(vmin_raw, 0.1)
        elif vmax_raw <= 3:
            # Caso para ET/P Ratio (0 a 1)
            vmin = np.floor(vmin_raw * 10) / 10
        else:
            # Caso para Precipitação/ET (números grandes)
            vmin = np.floor(vmin_raw / 5) * 5
            vmin = int(vmin)
            
        vmin = float(vmin)
        print('vmin', vmin)

    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    
    for key, data in data_dict.items():          
            var_name, period_value = key
            
            if period == 'monthly':
                if isinstance(period_value, datetime):
                    title_period = period_value.strftime("%m/%Y")
                    file_period = period_value.strftime("%Y_%m")
                elif isinstance(period_value, (tuple, list)):
                    year, month = period_value
                    title_period = f"{month:02d}/{year}"
                    file_period = f"{year}_{month:02d}"
                else:
                    title_period = f"{period_value:02d}"
                    file_period = f"Year_{period_value:02d}"
            
            elif period == 'annual':
                title_period = f"{period_value}"
                file_period = f"year_{period_value}"
                
            elif period == 'climatology':
                title_period = f"Climatology - Month {period_value:02d}"
                file_period = f"clim_{period_value:02d}"
    
            title = f"{title_prefix} - {title_period}"
            fname = f"{filename_prefix}_{file_period}.png"

            # ===== GRID =====
            fig, ax = plt.subplots(figsize=(9, 8))
            if is_grid:
                if zero_gray:
                    # apenas zeros viram NaN (para ficarem cinza)
                    data_plot = np.where(data == 0, 0, data)
                else:
                    data_plot = data
    
                data_masked = np.ma.masked_invalid(data_plot)
                valid_mask = ~np.isnan(data)
    
                if not np.any(valid_mask):
                    continue
                
                extent = [
                    longitude.min(), longitude.max(),
                    latitude.min(), latitude.max()
                ]
                
                if zero_gray:
                    cmap_obj = plt.get_cmap(cmap).copy()
                    cmap_obj.set_bad('lightgray')
                else:
                    cmap_obj = cmap
                    
                im = ax.imshow(
                    data_masked,
                    cmap=cmap_obj,
                    origin='lower',
                    extent=extent,
                    vmin=None if log_scale else vmin,
                    vmax=None if log_scale else vmax,
                    norm=norm,
                    zorder=1
                )
                
                # --- ADICIONAR MAPA DE PORTUGAL NO FUNDO ---
                try:
                    # ctx.providers.OpenStreetMap.Mapnik é gratuito e bom
                    # Se quiser estilo satélite: ctx.providers.Esri.WorldImagery
                    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
                except Exception as e:
                    print(f"⚠️ Erro ao carregar mapa de fundo: {e}")                   
                    
                if drainage_gdf is not None:
                    drainage_gdf.plot(
                        ax=ax,
                        linewidth=0.5,
                        color='lightgray',
                        alpha=0.6,
                        zorder=10
                    )
                
                if basins_gdf is not None and not isinstance(basins_gdf, bool):
                    basins_gdf.boundary.plot(
                        ax=ax,
                        linewidth=1.2,
                        color='black',
                        alpha=0.3,
                        zorder=20
                    ) 
                    
                cbar = fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.55, aspect=25)
                
            # ===== FLOW / REACHES =====
            else:
                for i in range(len(data)):
                    val = data[i]
                    if np.isnan(val) or val < flow_min:
                        color = 'lightgrey'
                    else:
                        color = cmap(norm(val))
    
                    ax.plot(
                        [x_nodes[up_nodes[i]], x_nodes[down_nodes[i]]],
                        [y_nodes[up_nodes[i]], y_nodes[down_nodes[i]]],
                        color=color,
                        linewidth=2
                    )
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label=cbar_label)
                ax.axis('equal')
            
            ax.set_title(title)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            ax.set_ylim(lat1, lat2)
            ax.set_xlim(lon1, lon2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.close('all')
            gc.collect()       
        
def compute_monthly(acc_daily, spatial_avg_daily, dates, accumulate_vars, mean_vars):
    """Agrega listas diárias em grids/médias mensais, mantendo NaN quando todos os dias forem NaN."""

    monthly_acc = defaultdict(list)
    monthly_spatial = defaultdict(list)
    # Criamos uma lista única com tudo o que deve ser processado
    valid_keys = set(accumulate_vars + mean_vars)
    for i, date in enumerate(dates):
        ym = (date.year, date.month)
        # CORREÇÃO AQUI: Itera apenas sobre as chaves válidas, não sobre tudo o que está no dict
        for key in valid_keys:
            if key in acc_daily:
                monthly_acc[(key, ym)].append(acc_daily[key][i])
            
            if key in spatial_avg_daily:
                monthly_spatial[(key, ym)].append(spatial_avg_daily[key][i])
    monthly_acc_final = {}
    monthly_spatial_final = {}
    for (key, ym), grid_list in monthly_acc.items():
        grid_stack = np.stack(grid_list)
        # acumulados
        if key in accumulate_vars:
            monthly_acc_final[(key, ym)] = np.where(
                np.all(np.isnan(grid_stack), axis=0),
                np.nan,
                np.nansum(grid_stack, axis=0)
            )
        # médias
        else:
            monthly_acc_final[(key, ym)] = np.where(
                np.all(np.isnan(grid_stack), axis=0),
                np.nan,
                np.nanmean(grid_stack, axis=0)
            )
    for (key, ym), val_list in monthly_spatial.items():
        val_array = np.array(val_list)
        if key in ["P", "Pobs", "ET", "RO", "Recharge", "Deficiency","Infiltration"]:
            monthly_spatial_final[(key, ym)] = np.nan if np.all(np.isnan(val_array)) else np.nansum(val_array)
        else:
            monthly_spatial_final[(key, ym)] = np.nan if np.all(np.isnan(val_array)) else np.nanmean(val_array)
    return monthly_acc_final, monthly_spatial_final

def compute_annual(acc_daily, spatial_avg_daily, dates,accumulate_vars, mean_vars):
    """Agrega listas diárias em grids/médias anuais, mantendo NaN quando todos os dias forem NaN."""
    
    annual_acc = defaultdict(list)
    annual_spatial = defaultdict(list)
    # Criamos o conjunto de chaves que REALMENTE queremos processar
    valid_keys = set(accumulate_vars + mean_vars)
    for i, date in enumerate(dates):
        year = date.year
        for key in valid_keys:
            # Verifica se a chave existe nos dicionários de entrada
            if key in acc_daily:
                annual_acc[(key, year)].append(acc_daily[key][i])
            
            if key in spatial_avg_daily:
                annual_spatial[(key, year)].append(spatial_avg_daily[key][i])
    annual_acc_final = {}
    annual_spatial_final = {}
    for (key, year), grid_list in annual_acc.items():
        grid_stack = np.stack(grid_list)
        if key in accumulate_vars:
            annual_acc_final[(key, year)] = np.where(
                np.all(np.isnan(grid_stack), axis=0),
                np.nan,
                np.nansum(grid_stack, axis=0)
            )
        else:
            annual_acc_final[(key, year)] = np.where(
                np.all(np.isnan(grid_stack), axis=0),
                np.nan,
                np.nanmean(grid_stack, axis=0)
            )
    for (key, year), val_list in annual_spatial.items():
        val_array = np.array(val_list)
        if key in ["P", "Pobs", "ET", "RO", "Recharge", "Deficiency", "Infiltration"]:
            annual_spatial_final[(key, year)] = np.nan if np.all(np.isnan(val_array)) else np.nansum(val_array)
        else:
            annual_spatial_final[(key, year)] = np.nan if np.all(np.isnan(val_array)) else np.nanmean(val_array)
    return annual_acc_final, annual_spatial_final
    
def compute_climatology(acc_daily, spatial_avg_daily, dates,
                        accumulate_vars, mean_vars):

    # 1. Definir chaves válidas (apenas as que têm dados e foram filtradas)
    valid_keys = set(accumulate_vars + mean_vars)
    # Agrupar dados por Ano e Mês para criar o "Total Mensal" primeiro
    monthly_totals_acc = defaultdict(lambda: defaultdict(list))
    monthly_totals_spatial = defaultdict(lambda: defaultdict(list))
    for i, date in enumerate(dates):
        year_month = (date.year, date.month)
        for key in valid_keys:
            if key in acc_daily:
                monthly_totals_acc[key][year_month].append(acc_daily[key][i])
            if key in spatial_avg_daily:
                monthly_totals_spatial[key][year_month].append(spatial_avg_daily[key][i])
    # 2. Agora consolidamos os meses
    climat_acc = defaultdict(list)
    climat_spatial = defaultdict(list)

    # --- GRID ---
    for key, year_month_dict in monthly_totals_acc.items():
        for (year, month), days_list in year_month_dict.items():
            stacked = np.stack(days_list)
            if key in accumulate_vars:
                # Total acumulado deste mês específico (ex: Soma de todos os dias de Jan/2011)
                res = np.nansum(stacked, axis=0)
            else:
                # Média deste mês específico
                res = np.nanmean(stacked, axis=0)
            climat_acc[(key, month)].append(res)

    # --- MÉDIA ESPACIAL ---
    for key, year_month_dict in monthly_totals_spatial.items():
        for (year, month), days_list in year_month_dict.items():
            vals = np.array(days_list)
            if key in accumulate_vars:
                monthly_val = np.nansum(vals)
            else:
                monthly_val = np.nanmean(vals)
            climat_spatial[(key, month)].append(monthly_val)

    # 3. Média Final da Climatologia (Média de todos os Janeiros, todos os Fevereiros...)
    climat_acc_final = {}
    climat_spatial_final = {}
    for (key, month), month_values_list in climat_acc.items():
        # Usando stack para tirar a média/mediana temporal dos grids mensais
        stacked_months = np.stack(month_values_list)
        res = np.nanmean(stacked_months, axis=0) # Alterado para mean para ser coerente com balanços
        
        if 'mask_nan' in globals() and mask_nan is not None and res.ndim == 2:
            if res.shape == mask_nan.shape:
                res = np.where(mask_nan, np.nan, res)
            elif res.shape == mask_nan.T.shape:
                res = np.where(mask_nan.T, np.nan, res)
        climat_acc_final[(key, month)] = res
    for (key, month), month_values_list in climat_spatial.items():
        climat_spatial_final[(key, month)] = np.nanmean(month_values_list)

    return climat_acc_final, climat_spatial_final

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
        # ========================= FLOW =======================
        # ======================================================
        # parâmetro que temos a média!
        hdf_flow = os.path.join(folder_path, "DrainageNetwork_5.hdf5")
        if os.path.exists(hdf_flow):
            with h5py.File(hdf_flow, 'r') as f:
                # ----------------------------------------------
                # Inicializar coordenadas dos reaches (uma vez só)
                # ----------------------------------------------
                if reach_centers is None:
                    x_nodes = f['Nodes/X'][:]
                    y_nodes = f['Nodes/Y'][:]
                    up_nodes   = f['Reaches/Up'][:].astype(int) - 1
                    down_nodes = f['Reaches/Down'][:].astype(int) - 1
                    reach_centers = np.column_stack((
                        (x_nodes[up_nodes] + x_nodes[down_nodes]) / 2,
                        (y_nodes[up_nodes] + y_nodes[down_nodes]) / 2
                    ))
                flow_data = np.array([f[f'Results/channel flow/{k}'][:] for k in f['Results/channel flow']])
                daily_mean_flow = np.mean(flow_data, axis=0)
                acc["FLOW"].append(daily_mean_flow)
                
        # ======================================================
        # ================== POROUS MEDIA (WTD) ================
        # ======================================================
        # parâmetro que temos a média!
        hdf_pm = os.path.join(folder_path, "PorousMedia_5.hdf5")
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
        hdf_evtp = os.path.join(folder_path, "EVTP_5.hdf5")
        if os.path.exists(hdf_evtp):
            with h5py.File(hdf_evtp, 'r') as f:
                daily_data = []
                et_group = f['Results/AccEVTP']
                daily_data = load_and_process_group(et_group, mask_nan=mask_nan)
                daily_data = daily_data.astype(np.float32)
                daily_sum = np.nansum(daily_data, axis=0)
                if mask_nan is not None:
                    daily_sum = np.where(mask_nan, np.nan, daily_sum)
                acc["ET"].append(daily_sum)
                spatial_avg["ET"].append(np.nanmean(daily_sum))

        # ======================================================
        # ======================== RUNOFF ======================
        # ======================================================
        # TRANSFORMAR PARA MM #
        hdf_ro = os.path.join(folder_path, "Runoff_5.hdf5")
        if os.path.exists(hdf_ro):
            with h5py.File(hdf_ro, 'r') as f:
                ro_group = f['Results/flow modulus']
                daily_data = load_and_process_group(ro_group, mask_nan=mask_nan)
                daily_data = daily_data.astype(np.float32)             
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
                
        # ======================================================
        # ======================== INFILT ======================
        # ======================================================
        hdf_basin = os.path.join(folder_path, "Basin_5.hdf5")
        if os.path.exists(hdf_evtp):
            with h5py.File(hdf_basin, 'r') as f:
                daily_data = []
                bs_group = f['Results/AccInfiltration']
                daily_data = load_and_process_group(bs_group, mask_nan=mask_nan)
                daily_total = daily_data[-1] - daily_data[0]
                daily_total_mm = np.where(np.isnan(daily_total), np.nan, daily_total * 1000)
            acc["Infiltration"].append(daily_total_mm)
            spatial_avg["Infiltration"].append(np.nanmean(daily_total_mm))
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
        print('ERA5:', file_date)
        
        with h5py.File(era_path, 'r') as f:
            ym = (file_date.year, file_date.month)
            
            # ---------- PRECIPITATION ----------
            if "Results/precipitation" not in f:
                print(f"⚠️ Variável 'precipitation' não encontrada em {filename}. Pulando arquivo.")
                continue
            daily_data = []
            p_group = f['Results/precipitation']
            keys = list(p_group.keys())
            for k in keys:
                data = p_group[k][:]  # shape: (180, 350)
                data = np.where(data < -99, np.nan, data)
                daily_data.append(data)
            daily_data = np.rot90(daily_data, k=1, axes=(1,2))
            daily_data = np.flip(daily_data, axis=1)
            if mask_nan is not None:
                daily_data = np.where(mask_nan, np.nan, daily_data)
            daily_sum = np.where(
                np.all(np.isnan(daily_data), axis=0),
                np.nan,
                np.nansum(daily_data, axis=0)
            ) 
            acc["P"].append(daily_sum)
            spatial_avg["P"].append(np.nanmean(daily_sum))
            
            # ---------- AIR TEMPERATURE ----------    
            daily_data = []
            t_group = f['Results/air temperature']  # acesso rápido
            keys = list(t_group.keys())
            for k in keys:
                data = t_group[k][:]  # shape: (180, 350)
                data = np.where(data < -99, np.nan, data)
                daily_data.append(data)

            daily_data = np.rot90(daily_data, k=1, axes=(1,2))
            daily_data = np.flip(daily_data, axis=1)
            
            daily_data = np.where(mask_nan, np.nan, daily_data)
                
            daily_mean = np.where(
                np.all(np.isnan(daily_data), axis=0),
                np.nan,
                np.nanmean(daily_data, axis=0)
            )  
            acc["T"].append(daily_mean)
            spatial_avg["T"].append(np.nanmean(daily_mean))
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
            print('OBS:', file_date)
            # ---------- PRECIPITAÇÃO DIÁRIA ----------
            data = p_group[pk][:]  # (ny, nx)
            data = np.where(data < 0, np.nan, data)
            data = np.rot90(data, k=1)
            data = np.flipud(data)
            data = np.where(mask_nan, np.nan, data)
            
            acc["Pobs"].append(data)
            spatial_avg["Pobs"].append(np.nanmean(data))
    print('✅ Observational Precipitation done')

# ------------------- CHECAR TAMANHOS -------------------
print("===== Comprimento das listas em acc =====")
for k, v in acc.items():
    print(f"{k}: {len(v)}")

print("\n===== Comprimento das listas em spatial_avg =====")
for k, v in spatial_avg.items():
    print(f"{k}: {len(v)}")

# Verificar se todas as listas em acc têm o mesmo tamanho
acc_lens = [len(v) for v in acc.values()]
print("\nTodas as listas em acc têm o mesmo tamanho?", all(l == acc_lens[0] for l in acc_lens))

# Verificar se todas as listas em spatial_avg têm o mesmo tamanho
spatial_lens = [len(v) for v in spatial_avg.values()]
print("Todas as listas em spatial_avg têm o mesmo tamanho?", all(l == spatial_lens[0] for l in spatial_lens))

# ======================================================
# ====================== WATER BALANCE =================
# ======================================================
    
# Verifica se Pobs tem dados, senão usa P (ERA5)
if "Pobs" in acc and len(acc["Pobs"]) > 0:
    P_source = "Pobs"
    print("ℹ️ Usando Precipitação Observacional")
else:
    P_source = "P"
    print("ℹ️ Usando Precipitação ERA5 (Pobs vazia ou ausente)")
    
limit = len(acc[P_source])
dates_list = [start_date + timedelta(days=i) for i in range(limit)]

P_dict  = {dates_list[i]: acc[P_source][i] for i in range(limit)}
ET_dict = {dates_list[i]: acc["ET"][i]     for i in range(limit)}
RO_dict = {dates_list[i]: acc["RO"][i]     for i in range(limit)}

recharge_daily, deficiency_daily, ratio_etp_daily = compute_water_balance(P_dict, ET_dict, RO_dict)

# Atualizar dicionários acumulados e médias espaciais
acc["Recharge"]   = [recharge_daily[d] for d in dates_list]
acc["Deficiency"] = [deficiency_daily[d] for d in dates_list]
acc["ETP_ratio"]  = [ratio_etp_daily[d] for d in dates_list]

for k, daily_dict in zip(["Recharge","Deficiency","ETP_ratio"],
                         [recharge_daily, deficiency_daily, ratio_etp_daily]):
    spatial_avg[k] = [np.nanmean(grid) for grid in daily_dict.values()]

# ======================================================
# ================== SUMMARY DATAFRAME =================
# ======================================================
data_dict = {
    "date": dates_list,
    "P": spatial_avg.get("P", [np.nan]*len(dates_list)),
    "T": spatial_avg.get("T", [np.nan]*len(dates_list)),
    #"SR": spatial_avg.get("SR", [np.nan]*len(dates_list)),
    "PM_mean": spatial_avg.get("PM", [np.nan]*len(dates_list)),
    "ET_sum": spatial_avg.get("ET", [np.nan]*len(dates_list)),
    "RO_mm": spatial_avg.get("RO", [np.nan]*len(dates_list)),
    "RO_flow_m3s": spatial_avg.get("RO_FLOW", [np.nan]*len(dates_list)),
    "Recharge": spatial_avg.get("Recharge", [np.nan]*len(dates_list)),
    "Infiltration": spatial_avg.get("Infiltration", [np.nan]*len(dates_list)),
    "Deficiency": spatial_avg.get("Deficiency", [np.nan]*len(dates_list)),
    "ETP_ratio": spatial_avg.get("ETP_ratio", [np.nan]*len(dates_list))
}

if "Pobs" in spatial_avg and len(spatial_avg["Pobs"]) == len(dates_list):
    data_dict["Pobs"] = spatial_avg["Pobs"]
else:
    data_dict["Pobs"] = [np.nan] * len(dates_list)

daily_df = pd.DataFrame(data_dict)
daily_df.to_csv(os.path.join("daily_values.csv"), index=False)
print("✅ Daily spatial averages saved to CSV")

# ======================================================
# ========= MONTH ANNUALLY AND CLIMATOLOGY =============
# ======================================================
accumulate_vars = ["P", "Pobs", "SR", "ET", "RO", "Recharge", "Deficiency", "Infiltration"]
mean_vars = ["PM", "RO_FLOW", "ETP_ratio", "T"]

accumulate_vars = [v for v in accumulate_vars if v in acc and len(acc[v]) > 0]
mean_vars = [v for v in mean_vars if v in acc and len(acc[v]) > 0]

acc_monthly, spatial_avg_monthly = compute_monthly(acc, spatial_avg, dates_list,accumulate_vars,mean_vars)
acc_annual, spatial_avg_annual   = compute_annual(acc, spatial_avg, dates_list,accumulate_vars,mean_vars)
acc_climat, spatial_avg_climat = compute_climatology(
    acc,
    spatial_avg,
    dates_list,
    accumulate_vars,
    mean_vars
)
print("✅ Monthly, Annual and Climatology dictionaries generated")

# ======================================================
# ======================== PLOTS =======================
# ======================================================
if plot == 1:
    variables = {
        "PM": {
            "title": "Water Table Depth (WTD)",
            "cbar": "WTD (m)",
            "prefix": "pm",
            "cmap": "turbo",
            "is_grid": True,
            "basins_gdf":False
        },

        "ET": {
            "title": "Evapotranspiration",
            "cbar": "ET (mm)",
            "prefix": "et",
            "cmap": "turbo",
            "is_grid": True
        },

        "RO": {
            "title": "Runoff",
            "cbar": "RO (mm)",
            "prefix": "ro",
            "cmap": "RdYlBu_r",
            "is_grid": True
        },
        
        "RO_FLOW": {
            "title": "Runoff",
            "cbar": r"Flow ($m^3 \cdot s^{-1}$)",
            "prefix": "flow_m3s",
            "cmap": "Blues",
            "is_grid": True
        },
        "Infiltration": {
            "title": "Infiltration",
            "cbar": "Accumulated Infiltration (mm)",
            "prefix": "Infiltration",
            "cmap": "RdYlBu_r",
            "is_grid": True
        },
        "Recharge": {
            "title": "Recharge",
            "cbar": "Recharge (mm)",
            "prefix": "recharge",
            "cmap": "RdYlBu_r",
            "is_grid": True
        },
        "ETP_ratio": {
            "title": "ET / P Ratio",
            "cbar": "ET/P",
            "prefix": "etp_ratio",
            "cmap": "coolwarm",
            "is_grid": True
        },
        
        "Pobs": {
            "title": "Accumulated precipitation",
            "cbar": "Accumulated precipitation (mm)",
            "prefix": "Precipitation",
            "cmap": "Blues",
            "is_grid": True
        },

        "P": {
            "title": "Accumulated precipitation ERA5",
            "cbar": "Accumulated precipitation (mm)",
            "prefix": "ERA5",
            "cmap": "Blues",
            "is_grid": True
        },
        
        "T": {
            "title": "Average temperature ERA5",
            "cbar": "Average temperature (ºC)",
            "prefix": "temperature",
            "cmap": "autumn_r",
            "is_grid": True
        } 
    }
    
    # ---------------- Outputs  ----------------
    output_dirs = {
        "monthly": "plots/monthly",
        "annual": "plots/annual",
        "climatology": "plots/climatology"
    }
    os.makedirs(output_dirs["monthly"], exist_ok=True)
    os.makedirs(output_dirs["annual"], exist_ok=True)
    os.makedirs(output_dirs["climatology"], exist_ok=True)

    # Loop para plotar todas as variáveis e escalas
    for var_name, var_props in variables.items():
        print(f"🚀 Iniciando processamento da variável: {var_name} ({var_props['title']})")
        for period, acc_dict in zip(
            ["monthly", "annual", "climatology"],
            [acc_monthly, acc_annual, acc_climat]
        ):
            print(f"   ∟ Gerando mapas para escala: {period}...")
            # Filtra apenas a variável desejada
            data_dict = {k: v for k, v in acc_dict.items() if k[0] == var_name}
            
            if len(data_dict) == 0:
                continue  
            
            plot_images(
                data_dict=data_dict,
                title_prefix=var_props["title"],
                cbar_label=var_props["cbar"],
                filename_prefix=var_props["prefix"],
                output_dir=output_dirs[period],
                cmap=var_props["cmap"],
                is_grid=var_props["is_grid"],
                log_scale=(var_name=="RO" or var_name=="flow"),
                basins_gdf = False
            )