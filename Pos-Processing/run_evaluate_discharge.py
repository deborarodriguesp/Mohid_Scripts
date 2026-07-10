import os,sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
import pymannkendall as mk

global all_stats
global mk_stats

def format_station_name(name):
    """Corrige nomes específicos ou formata o padrão."""
    name_lower = name.lower().strip()
    if name_lower in NAME_FIX:
        return NAME_FIX[name_lower]
    # Se não estiver no dicionário, troca _ por espaço e põe Capitalizado
    return name.replace("_", " ").title()

# ============================================================
# 2️⃣ FUNÇÕES DE MÉTRICAS
# ============================================================

def compute_stats(obs, sim):
    obs, sim = np.array(obs), np.array(sim)
    
    # Criar uma máscara para pegar apenas onde AMBOS existem
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]

    # Se após remover NaNs sobrar pouco dado (ex: menos de 30 dias), retorna vazio
    if len(o) < 10:
        return {"RMSE": np.nan, "NSE": np.nan, "R2": np.nan, "PBIAS": np.nan}

    # Cálculos usando apenas os dados válidos
    rmse = np.sqrt(np.mean((s - o)**2))
    
    mean_o = np.mean(o)
    rrmse = (rmse / mean_o) if mean_o != 0 else np.nan
    
    r_sq = np.corrcoef(o, s)[0, 1]**2
    nse = 1 - np.sum((s - o)**2) / np.sum((o - np.mean(o))**2)
    pbias = 100 * np.sum(s - o) / np.sum(o)

    return {
            "RMSE": round(rmse, 2), 
            "RRMSE": round(rrmse, 2), 
            "NSE": round(nse, 2), 
            "R2": round(r_sq, 2), 
            "PBIAS": int(round(pbias, 2))
        }
        
def mann_kendall_test(series):
    # Remove NaNs para o teste de tendência
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return {'trend': 'no trend', 'slope': 0, 'p': 1, 'z': 0}
    res = mk.original_test(clean_series.values)
    return {'trend': res.trend, 'slope': res.slope, 'p': res.p, 'z': res.z}

def sens_line(series, slope):
    series_clean = series.dropna()
    x = np.arange(len(series_clean))
    intercept = np.median(series_clean.values - slope * x)
    # Retorna a linha para o índice original (anos)
    return intercept + slope * np.arange(len(series))

# ============================================================
# 3️⃣ PROCESSAMENTO DE ARQUIVOS
# ============================================================
def load_obs_data(path):
    df = pd.read_csv(path, sep=';').rename(columns=lambda x: x.strip())
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df.set_index('Date')

obs_all = load_obs_data(path_obs)

def process_file(file_path):
    name = os.path.basename(file_path).replace(".csv", "")
    df_sim = pd.read_csv(file_path, sep=',', skipinitialspace=True).rename(columns=lambda x: x.strip())
    df_sim['Date'] = pd.to_datetime(df_sim['Date'], dayfirst=True, errors='coerce')
    df_sim.set_index('Date', inplace=True)
    df_sim = df_sim.loc[start_date:end_date]

    results_data = []
    
    if "Node" in name:
        raw_names = [(name.split("_Node_")[1].replace("_daily", ""), "channel_flow")]
    else:
        base = name.split("_Reservoir_")[1].replace("_daily", "")
        raw_names = [(f"{base}_inflow", "Inflow"), (f"{base}_outflow", "Outflow")]

    for raw_st_name, col_sim in raw_names:
        # Nome bonito para o gráfico
        pretty_name = format_station_name(raw_st_name)
        
        obs_col = raw_st_name.replace("_inflow", "_i").replace("_outflow", "_o")
        df_d = pd.DataFrame(index=pd.date_range(start_date, end_date, freq='D'))
        df_d['sim'] = df_sim[col_sim]
        df_d['obs'] = obs_all[obs_col] if obs_col in obs_all.columns else np.nan

        df_m = df_d.resample("M").mean()
        #df_y = df_d.resample("Y").mean()
        df_y_obs = df_d['obs'].resample("Y").apply(lambda x: x.mean() if x.count() > 300 else np.nan)
        df_y_sim = df_d['sim'].resample("Y").mean() # O sim sempre tem tudo

        df_y = pd.DataFrame({'obs': df_y_obs, 'sim': df_y_sim})  
        
        # --- CÁLCULO DE ESTATÍSTICAS (Diário e Mensal) ---
        for scale_name, df_work in [("Daily", df_d), ("Monthly", df_m)]:
            # Calibração
            s_cal = compute_stats(df_work.loc[cal_start:cal_end, "obs"], df_work.loc[cal_start:cal_end, "sim"])
            # Validação
            s_val = compute_stats(df_work.loc[val_start:val_end, "obs"], df_work.loc[val_start:val_end, "sim"])
            
            all_stats.append({
                "Station": pretty_name,
                "Scale": scale_name,
                "RMSE_cal": round(s_cal["RMSE"], 2), "RMSE_val": round(s_val["RMSE"], 2),
                "RRMSE_cal": s_cal["RRMSE"], "RRMSE_val": s_val["RRMSE"],
                "NSE_cal": round(s_cal["NSE"], 3), "NSE_val": round(s_val["NSE"], 3),
                "R2_cal": round(s_cal["R2"], 3),   "R2_val": round(s_val["R2"], 3),
                "PBIAS_cal": round(s_cal["PBIAS"], 2), "PBIAS_val": round(s_val["PBIAS"], 2)
            })
        
        # --- MANN-KENDALL (Anual) ---
        mk_o = mann_kendall_test(df_y['obs'])
        mk_s = mann_kendall_test(df_y['sim'])
        
        # Statistics
        m_cal = compute_stats(df_m.loc[cal_start:cal_end, "obs"], df_m.loc[cal_start:cal_end, "sim"])
        m_val = compute_stats(df_m.loc[val_start:val_end, "obs"], df_m.loc[val_start:val_end, "sim"])
        
        # Médias Anuais
        avg_q_obs = df_y['obs'].mean()
        avg_q_sim = df_y['sim'].mean()
        
        # Cálculo de Redução (%) ANUAL
        # (Slope * n_anos) / Média * 100
        red_obs_perc = (mk_o['slope'] / avg_q_obs * 100) if avg_q_obs > 0 else 0
        red_sim_perc = (mk_s['slope'] / avg_q_sim * 100) if avg_q_sim > 0 else 0
        
        # Cálculo de Redução (%) no período (10 anos)
        # (Slope * n_anos) / Média * 100
        red10_obs_perc = (mk_o['slope'] * 10 / avg_q_obs * 100) if avg_q_obs > 0 else 0
        red10_sim_perc = (mk_s['slope'] * 10 / avg_q_sim * 100) if avg_q_sim > 0 else 0
        
        # TABELA COMPLETA MK
        mk_stats_list.append({
            "Station": pretty_name,
            "trend_obs": mk_o['trend'],
            "Z_obs": round(mk_o['z'], 2),
            "p_obs": round(mk_o['p'], 3),
            "slope_obs": round(mk_o['slope'], 2),
            "Annual_Q_obs": round(avg_q_obs, 1),
            "Reduction_obs(%)": round(red_obs_perc, 1),
            "Reduction_obs_10y(%)": round(red10_obs_perc, 1),
            "trend_sim": mk_s['trend'],
            "Z_sim": round(mk_s['z'], 2),
            "p_sim": round(mk_s['p'], 3),
            "slope_sim": round(mk_s['slope'], 2),
            "Annual_Q_sim": round(avg_q_sim, 1),
            "Reduction_sim(%)": round(red_sim_perc, 1),
            "Reduction_sim_10y(%)": round(red10_sim_perc, 1)
        })
        results_data.append({
            "station": pretty_name, "df_m": df_m, "df_y": df_y, "mk_o": mk_o, "mk_s": mk_s
        })
    return results_data

# ============================================================
# 4️⃣ FUNÇÃO DO ATLAS (TODOS SUBPLOTS EM UMA IMAGEM)
# ============================================================
def generate_discharge_atlas(data_list, basin_label, output_dir):
    if not data_list: return
    n = len(data_list)
    
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(n, 1, figsize=(16, 5*n), sharex=True)
    if n == 1: axes = [axes]
    
    split_date = pd.Timestamp("2015-01-01")
    
    for i, item in enumerate(data_list):
        ax = axes[i]
        df, st = item['df_m'], item['station']
        
        # Preenchimento do Viés (Sombra cinza entre as linhas)
        ax.fill_between(df.index, df["obs"], df["sim"], 
                        color='gray', alpha=0.2, label="Bias (Difference)")
                        
        # Linhas principais. O observado (preto) terá falhas onde for NaN
        ax.plot(df.index, df["obs"], color='black', label="Observed", linewidth=2.0, zorder=3)
        ax.plot(df.index, df["sim"], color='red', label="Simulated", linewidth=2, zorder=3)
        
        #Estética do gráfico
        ax.axvline(split_date, color='gray', linestyle='--', linewidth=2, zorder=1)
        ax.set_title(f"Station: {item['station']}", fontsize=18, fontweight='bold', loc='left')
        ax.set_ylabel(r"Q ($\mathbf{m^3 \cdot s^{-1}}$)", fontweight='bold', fontsize=20)
        ax.grid(alpha=0.3)
        
        # Labels de Calibração/Validação apenas no primeiro gráfico para não poluir
        if i == 0:
            y_max = max(df["sim"].max(), df["obs"].max() if not df["obs"].isnull().all() else df["sim"].max())
            ax.text(pd.Timestamp("2013-01-01"), y_max*0.9, "Calibration", color='gray', fontsize=16, ha='center', fontweight='bold')
            ax.text(pd.Timestamp("2018-01-01"), y_max*0.9, "Validation", color='gray', fontsize=16, ha='center', fontweight='bold')

    plt.xlabel("Year", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(output_dir, f"Atlas_Discharge_{basin_label}.png"), dpi=200)
    plt.close()
    
def generate_mk_atlas(data_list, basin_label, output_dir):
    """Gera Atlas de tendências anuais com Sen's Slope"""
    n = len(data_list)
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(n, 1, figsize=(16, 5*n), sharex=True)
    if n == 1: axes = [axes]
    
    for i, item in enumerate(data_list):
        ax = axes[i]
        df_y = item['df_y']
        years = df_y.index.year
        
        # Plot das séries anuais
        ax.plot(years, df_y["obs"], color='black', marker='o', label="Observed Mean", linewidth=2)
        ax.plot(years, df_y["sim"], color='red', marker='s', label="Simulated Mean", linewidth=2)
        
        # Linhas de Tendência (Sen's Slope)
        obs_sen = sens_line(df_y["obs"], item['mk_o']['slope'])
        sim_sen = sens_line(df_y["sim"], item['mk_s']['slope'])
        
        ax.plot(years, obs_sen, color='black', linestyle='--', alpha=0.7, label=f"Obs Slope: {item['mk_o']['slope']:.2f}")
        ax.plot(years, sim_sen, color='red', linestyle='--', alpha=0.7, label=f"Sim Slope: {item['mk_s']['slope']:.2f}")
        
        ax.set_title(f"Trend Analysis: {item['station']}", fontsize=18, fontweight='bold', loc='left')
        ax.set_ylabel(r"Annual Q ($\mathbf{m^3 \cdot s^{-1}}$)", fontsize=16, fontweight='bold')
        ax.grid(alpha=0.2)
        ax.legend(loc='upper right', ncol=2, fontsize=12)

    plt.xlabel("Year", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Atlas_MK_Trends_{basin_label}.png"), dpi=200)
    plt.close()

def generate_combined_atlas(data_list, basin_label, output_dir):
    """Gera um Atlas único: Descarga (Esq) e MK Trends (Dir) por estação."""
    if not data_list: return
    n = len(data_list)
    
    plt.rcParams.update({'font.size': 20})
    
    # Criamos a figura. Width_ratios faz o MK ser mais estreito (2.5 para 1)
    fig = plt.figure(figsize=(24, 5 * n))
    gs = GridSpec(n, 2, width_ratios=[2.5, 1], figure=fig)
    
    split_date = pd.Timestamp("2015-01-01")
    border_width = 2.0
    
    for i, item in enumerate(data_list):
        # --- COLUNA 1: DESCARGA MENSAL ---
        ax_q = fig.add_subplot(gs[i, 0])
        df_m, st = item['df_m'], item['station']
        # Engrossando as bordas do gráfico de descarga
        for spine in ax_q.spines.values():
            spine.set_linewidth(border_width)
        ax_q.text(-0.11, 0.5, st, 
                 transform=ax_q.transAxes, 
                 fontsize=24, 
                 fontweight='bold', 
                 va='center', 
                 ha='center', 
                 rotation=90)    # Rotação para ler de baixo para cima
                 #bbox=dict(facecolor='lightgray', alpha=0.2, edgecolor='none', pad=10)
                 
        
        
        ax_q.plot(df_m.index, df_m["obs"], color='black', label="Observed", linewidth=2, zorder=3)
        ax_q.plot(df_m.index, df_m["sim"], color='red', label="Simulated", linewidth=2, alpha=1, zorder=3)
        ax_q.fill_between(df_m.index, df_m["obs"], df_m["sim"], color='gray', alpha=0.5, label="Bias")
        
        #ax_q.set_title(f"Station: {st}", fontsize=18, fontweight='bold', loc='left')
        ax_q.set_ylabel(r"Q ($\mathrm{mm \cdot yr^{-1}}$)", fontsize=20)
        ax_q.grid(alpha=0.3)
        
        ax_q.axvline(split_date, color='dimgray', linestyle='--', linewidth=2, alpha=1)
        if i == 0:
            y_max = max(df_m["sim"].max(), df_m["obs"].max() if not df_m["obs"].isnull().all() else df_m["sim"].max())
            ax_q.text(pd.Timestamp("2013-01-01"), y_max*0.9, "Calibration", color='black', fontsize=20, ha='center', alpha=1, fontweight='bold')
            ax_q.text(pd.Timestamp("2018-01-01"), y_max*0.9, "Validation", color='black', fontsize=20, ha='center', alpha=1, fontweight='bold')

        # --- COLUNA 2: MANN-KENDALL ANUAL ---
        ax_mk = fig.add_subplot(gs[i, 1])
        for spine in ax_mk.spines.values():
            spine.set_linewidth(border_width)
            
        df_y = item['df_y']
        years = df_y.index.year
        
        ax_mk.plot(years, df_y["obs"], color='black', marker='o', markersize=4, label="Obs. Annual", linewidth=2)
        ax_mk.plot(years, df_y["sim"], color='red', marker='s', markersize=4, label="Sim. Annual", linewidth=2)
        
        obs_sen = sens_line(df_y["obs"], item['mk_o']['slope'])
        sim_sen = sens_line(df_y["sim"], item['mk_s']['slope'])
        ax_mk.plot(years, obs_sen, color='black', linestyle='--',linewidth=2)
        ax_mk.plot(years, sim_sen, color='red', linestyle='--',linewidth=2)
        
        ax_mk.set_title(f"Trend Analysis", fontsize=20, loc='right')
        ax_mk.grid(alpha=0.2)
        
        # Pequena caixa de texto com os Slopes
        ax_mk.text(0.05, 0.05, f"Slope Obs: {item['mk_o']['slope']:.0f}\nSlope Sim: {item['mk_s']['slope']:.0f}", 
                   transform=ax_mk.transAxes, fontsize=20, bbox=dict(facecolor='white', alpha=0.7))

    # Legenda única no topo ou base
    handles, labels = ax_q.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=20)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(os.path.join(output_dir, f"Atlas_Combined_{basin_label}.png"), dpi=300)
    plt.close()

def sort_results(results, order_list):
    # Cria um mapa de posição para busca rápida
    pos_map = {name: i for i, name in enumerate(order_list)}
    # Retorna a lista ordenada. Se a estação não estiver na lista, vai para o final.
    return sorted(results, key=lambda x: pos_map.get(x['station'], 999))

def save_ordered_performance(stats_list, output_path):
    # 1. Transformar em DataFrame
    df = pd.DataFrame(stats_list)
    
    # 2. Converter colunas de escala e estação em 'Categorical' para ordenar conforme nossas listas
    df['Scale'] = pd.Categorical(df['Scale'], categories=ordem_escala, ordered=True)
    df['Station'] = pd.Categorical(df['Station'], categories=ordem_estacoes, ordered=True)
    
    # 3. Ordenar: Primeiro por Estação (para manter Araguaia e Tocantins em blocos) 
    # e depois por Escala.
    # Nota: Como sua lista de estações já segue a lógica Araguaia -> Tocantins, 
    # ordenamos por Estação primeiro e depois Escala.
    
    df_sorted = df.sort_values(by=['Station', 'Scale'])
    
    # 4. Reorganizar colunas para a ordem exata que você pediu
    cols_order = [
        'Station', 'Scale', 
        'RMSE_cal', 'RRMSE_cal', 'NSE_cal', 'R2_cal', 'PBIAS_cal',
        'RMSE_val', 'RRMSE_val', 'NSE_val', 'R2_val', 'PBIAS_val'
    ]
    
    # Garantir que apenas colunas existentes sejam usadas (evita erro se alguma faltar)
    final_cols = [c for c in cols_order if c in df_sorted.columns]
    df_sorted = df_sorted[final_cols]

    # 5. Salvar
    df_sorted.to_excel(output_path, index=False)
    # Se preferir CSV: df_sorted.to_csv(output_path.replace(".xlsx", ".csv"), index=False, sep=';')
    
    print(f"✅ Tabela ordenada salva em: {output_path}")
    
# ============================================================
# 1️⃣ CAMINHOS
# ============================================================

path_obs = r"E:\Modelos\WaterBalance_TAW\Descarga\Descarga_Obs.txt"
path_araguaia = r"E:\Modelos\WaterBalance_TAW\Descarga\Descarga_Araguaia"
path_tocantins = r"E:\Modelos\WaterBalance_TAW\Descarga\Descarga_Tocantins"

output_dir = r"E:\Modelos\WaterBalance_TAW\Descarga\outputs"
os.makedirs(output_dir, exist_ok=True)

start_date, end_date = "2011-01-01", "2020-12-31"
cal_start, cal_end = "2011-01-01", "2015-01-01"
val_start, val_end = "2015-01-01", "2020-12-31"

NAME_FIX = {
    "aruana": "Aruanã",
    "xambioa": "Xambioá",
    "torixoreu": "Torixoréu",
    "peixeangical_inflow": "Peixe Angical Inflow",
    "peixeangical_outflow": "Peixe Angical Outflow",
    "serradamesa_inflow": "Serra da Mesa Inflow",
    "serradamesa_outflow": "Serra da Mesa Outflow"
}

# ============================================================
# 5️⃣ EXECUÇÃO PRINCIPAL
# ============================================================
all_stats, mk_stats_list = [], []
araguaia_results, tocantins_results = [], []

print("🚀 Processando bacias...")
for f in glob.glob(os.path.join(path_araguaia, "*.csv")): araguaia_results.extend(process_file(f))
for f in glob.glob(os.path.join(path_tocantins, "*.csv")): tocantins_results.extend(process_file(f))

# --- LÓGICA DE ORDENAÇÃO ---
# Defina a ordem exata baseada nos Nomes Bonitos (pretty_name)
ordem_araguaia = [
    "Torixoréu", "Araguaiana", "Aruanã", 
    "Bandeirantes", "Xambioá", "Araguatins"
]
ordem_tocantins = [
    "Serra da Mesa Inflow", "Serra da Mesa Outflow", 
    "Peixe Angical Inflow", "Peixe Angical Outflow", "Tupiratins"
]

# Ordenando as listas antes de gerar os Atlas
araguaia_results = sort_results(araguaia_results, ordem_araguaia)
tocantins_results = sort_results(tocantins_results, ordem_tocantins)

print("🎨 Gerando Atlas Combinados...")
generate_combined_atlas(araguaia_results, "Araguaia", output_dir)
generate_combined_atlas(tocantins_results, "Tocantins", output_dir)

# Salvar Planilhas
pd.DataFrame(all_stats).to_excel(os.path.join(output_dir, "Performance_Summary.xlsx"), index=False)
pd.DataFrame(mk_stats_list).to_excel(os.path.join(output_dir, "MK_Trends_Summary.xlsx"), index=False)

print(f"✅ Concluído! Verifique a pasta: {output_dir}")

# Ordem geográfica desejada para as estações
ordem_estacoes = [
    "Torixoréu","Araguaiana", "Aruanã", "Bandeirantes", "Xambioá", "Araguatins",
    "Serra da Mesa Inflow", "Serra da Mesa Outflow", "Peixe Angical Inflow", "Peixe Angical Outflow", "Tupiratins"
]

# Ordem de escala (Daily primeiro, depois Monthly)
ordem_escala = ["Daily", "Monthly"]

# --- Na Execução Principal ---
save_ordered_performance(all_stats, os.path.join(output_dir, "Performance_Summary_Ordered.xlsx"))