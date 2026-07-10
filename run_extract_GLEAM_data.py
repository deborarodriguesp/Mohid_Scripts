import xarray as xr
import geopandas as gpd
import rioxarray
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --- DIRECTORY CONFIGURATIONS ---
base_dir = r"E:\METEOROLOGIA\GLEAM\anos" 
sub_folders = ["ET", "SMrz", "S"]
output_root = r"E:\METEOROLOGIA\GLEAM\resultados_consolidados"

shapefile_dir = "E:/Modelos/shapefiles/"
shapefiles_dict = {
    "estuary": "estuary/exterior_line.shp",
    "mortes": "mortes/mortes.shp",
    "alto_araguaia": "alto_araguaia/alto_araguaia.shp",
    "medio_araguaia" :"medio_araguaia/medio_araguaia.shp",
    "baixo_araguaia" :"baixo_araguaia/baixo_araguaia.shp",
    "araguaia": "araguaia/araguaia.shp",
    "sono" :"sono/sono.shp",
    "parana" :"parana/parana.shp",
    "medio_tocantins":"medio_tocantins/medio_tocantins.shp",
    "alto_tocantins": "alto_tocantins/alto_tocantins.shp",
    "tocantins" :"tocantins/tocantins.shp"
}

os.makedirs(output_root, exist_ok=True)

# Dictionary to store final tables grouped by parameter
final_tables = {var: pd.DataFrame() for var in sub_folders}

# --- 1. DATA PROCESSING ---
for var_folder in sub_folders:
    print(f"\n🚀 Starting processing for parameter: {var_folder}")
    netcdf_dir = os.path.join(base_dir, var_folder)
    netcdf_files = sorted(glob.glob(os.path.join(netcdf_dir, "*.nc")))

    if not netcdf_files:
        print(f"⚠️ No NetCDF files found in: {netcdf_dir}")
        continue

    # List to collect data across all basins for the current parameter
    basins_data_list = []

    for basin_id, rel_path in shapefiles_dict.items():
        shp_path = os.path.join(shapefile_dir, rel_path)
        if not os.path.exists(shp_path):
            print(f"⚠️ Shapefile not found for: {basin_id}")
            continue

        print(f"  📂 Extracting basin: {basin_id}")
        gdf = gpd.read_file(shp_path)
        
        # List to collect data across all years for this specific basin
        years_list = []

        for nc_path in netcdf_files:
            ds = xr.open_dataset(nc_path, chunks="auto")
            var_name = [v for v in ds.data_vars if v not in ['time', 'lat', 'lon']][0]
            
            # Spatial Setup and Clip Setup
            da = ds[var_name].rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            da = da.rio.write_crs("EPSG:4326")
            
            gdf = gdf.to_crs(da.rio.crs)

            try:
                da_clip = da.rio.clip(gdf.geometry, gdf.crs, drop=True, all_touched=True)
                df_temp = da_clip.mean(dim=["lat", "lon"]).to_dataframe().reset_index()
                
                # Retain only time and parameter values
                df_temp = df_temp[['time', var_name]]
                years_list.append(df_temp)
            except Exception as e:
                print(f"    ❌ Error clipping {basin_id}: {e}")
            ds.close()

        # Merge all years for the current basin and rename the column to the basin ID
        df_basin_full = pd.concat(years_list).set_index('time')
        df_basin_full.columns = [basin_id]
        basins_data_list.append(df_basin_full)

    # Merge all basins side-by-side for the current parameter
    if basins_data_list:
        final_tables[var_folder] = pd.concat(basins_data_list, axis=1)
        
        # Export the consolidated daily time-series CSV
        out_csv = os.path.join(output_root, f"Consolidated_Daily_{var_folder}.csv")
        final_tables[var_folder].to_csv(out_csv)
        print(f"✅ File saved successfully: {out_csv}")

# --- 2. CLIMATOLOGY GENERATION FOR JOURNAL MANUSCRIPT ---
print("\n📊 Generating Monthly Climatology for data visualization...")

for var, df in final_tables.items():
    if df.empty: 
        continue
    
    # Calculate historical monthly average across all basins simultaneously
    monthly_climatology = df.groupby(df.index.month).mean()
    monthly_climatology.index.name = 'Month'
    monthly_climatology.to_csv(os.path.join(output_root, f"Monthly_Climatology_{var}.csv"))
    
    # Generate regional comparison plot for this parameter
    plt.figure(figsize=(12, 6))
    for col in monthly_climatology.columns:
        plt.plot(monthly_climatology.index, monthly_climatology[col], marker='o', label=col)
    
    plt.title(f'Regional Comparison: {var}')
    plt.xlabel('Month')
    plt.ylabel(f'Mean Value ({var})')
    plt.xticks(range(1, 13))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, f"Comparative_Plot_{var}.png"), dpi=300)
    plt.close()

print(f"\n🚀 All operations completed successfully! Check outputs in: {output_root}")