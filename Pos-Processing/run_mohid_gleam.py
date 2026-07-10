import pandas as pd
import os, sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
from scipy.stats import theilslopes

def rmse(obs, sim):
    return np.sqrt(np.mean((sim - obs) ** 2))

def rrmse(obs, sim):
    return rmse(obs, sim) / np.mean(obs)

def nse(obs, sim):
    return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def pbias(obs, sim):
    return 100 * np.sum(sim - obs) / np.sum(obs)

def compute_stats(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    return {
        "RMSE": rmse(obs, sim),
        "RRMSE": rrmse(obs, sim),
        "NSE": nse(obs, sim),
        "R2": r**2,
        "PBIAS (%)": pbias(obs, sim)
    }
    
def aggregate_timescales(df, accumulate_vars, mean_vars, date_col="Date"):
    # --- Mensal ---
    df_monthly = (
        df
        .groupby(["Basin", df[date_col].dt.to_period("M")])
        .agg({**{var: "sum" for var in accumulate_vars},
              **{var: "mean" for var in mean_vars}})
        .reset_index()
    )
    # Transformar Period em Timestamp
    df_monthly = df_monthly.rename(columns={date_col: "YearMonth"})
    df_monthly["YearMonth"] = df_monthly["YearMonth"].dt.to_timestamp()
    
    # --- Anual ---
    df["Year"] = df[date_col].dt.year
    df_annual = (
        df
        .groupby(["Basin", "Year"])
        .agg({**{var: "sum" for var in accumulate_vars},
              **{var: "mean" for var in mean_vars}})
        .reset_index()
    )
    
    # --- Climatologia mensal ---
    df_climat = (
        df_monthly
        .assign(Month=lambda x: x["YearMonth"].dt.month)
        .groupby(["Basin", "Month"])
        .mean(numeric_only=True)
        .reset_index()
    )
    return df_monthly, df_annual, df_climat
    
def load_mohid_files(file_list, source_name):
    dfs = []
    for file in file_list:
        print(f"Lendo {file}")
        df = pd.read_csv(file, parse_dates=["Date"])
        df["source"] = source_name
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["DOY"] = df["Date"].dt.dayofyear
        dfs.append(df)
    df_daily = pd.concat(dfs, ignore_index=True)
    return df_daily
    
def load_gleam(files, var_names):
    dfs_long = []
    for file, var in zip(files, var_names):
        # Ler CSV
        df = pd.read_csv(file, sep=";", parse_dates=["time"], dayfirst=True)
        df = df.rename(columns={"time": "Date"})
        df_long = df.melt(
            id_vars="Date",
            var_name="Basin",
            value_name=var)
        # Criar colunas auxiliares
        df_long["Year"] = df_long["Date"].dt.year
        df_long["Month"] = df_long["Date"].dt.month
        df_long["DOY"] = df_long["Date"].dt.dayofyear
        dfs_long.append(df_long)
    # Merge horizontal de todos os arquivos
    df_final = dfs_long[0]
    for df_next in dfs_long[1:]:
        df_final = df_final.merge(df_next, on=["Date","Basin","Year","Month","DOY"])
    return df_final
    
def get_sorted_basins(df):
    # Lista ajustada para os nomes reais que aparecem no seu print
    desired_order = [
        "alto_tocantins",
        "parana",
        "sono",
        "medio_tocantins",
        "tocantins"
    ]
    
    # Pegamos as bacias que existem no dataframe
    existing_basins = df["Basin"].unique().tolist()
    
    # Filtramos para garantir que só tentaremos ordenar o que existe
    sorted_list = [b for b in desired_order if b in existing_basins]
    
    # Caso apareça alguma bacia nova/inesperada no futuro
    extra_basins = [b for b in existing_basins if b not in desired_order]
    
    return extra_basins + sorted_list    

# =================================================================
# ESTATÍSTICAS (DIÁRIO E MENSAL)
# =================================================================
def calculate_all_stats(df_m, df_g, time_col_m, time_col_g):
    # Merge para garantir comparação pareada
    df_stats = df_m.merge(df_g, left_on=["Basin", time_col_m], right_on=["Basin", time_col_g])
    
    # Garantir que temos uma coluna de ano para o filtro
    if "Year" not in df_stats.columns:
        df_stats["Year"] = df_stats[time_col_m].dt.year
        
    results = []
    # Usamos a nossa nova ordem de bacias!
    ordered_basins = get_sorted_basins(df_stats)
    
    for basin in ordered_basins:
        sub_basin = df_stats[df_stats["Basin"] == basin]
        
        # --- DIVISÃO DOS PERÍODOS ---
        periods = {
            "Calibration (2011-2014)": sub_basin[sub_basin["Year"].between(2011, 2014)],
            "Validation (2015-2021)": sub_basin[sub_basin["Year"].between(2015, 2021)],
            "Full Period": sub_basin
        }
        
        for p_name, sub in periods.items():
            if not sub.empty:
                # Stats para ET (MOHID vs GLEAM)
                s = compute_stats(sub["ET_gleam"].values, sub["ET"].values)
                s["Basin"] = basin
                s["Period"] = p_name
                results.append(s)
    
    return pd.DataFrame(results)

def plot_ET_monthly(df_m_monthly, df_g_monthly, output_dir):
    # 1. Preparação e Sincronização
    df_m = df_m_monthly.copy()
    df_g = df_g_monthly.copy()
    
    # Garantir nomes e chaves batendo
    df_m["Basin"] = df_m["Basin"].str.strip().str.lower()
    df_g["Basin"] = df_g["Basin"].str.strip().str.lower()
    
    # Cruzamento de dados
    df_plot = df_m.merge(df_g, on=["Basin", "YearMonth"])
    
    basins = get_sorted_basins(df_plot)
    n_basins = len(basins)
            
    border_width = 2.0
    plt.rcParams.update({'font.size': 20})
    
    fig, axes = plt.subplots(nrows=n_basins, ncols=1, figsize=(16, 4.5 * n_basins), sharex=True)
    if n_basins == 1: axes = [axes]
    
    # Definir a data de corte (início de 2015) para calibração e validação
    split_date = pd.Timestamp("2015-01-01")
    
    for i, basin in enumerate(basins):
        
        sub = df_plot[df_plot["Basin"] == basin].sort_values("YearMonth")
        basin_name = basin.replace('_', ' ').title()
        basin_name = basin_name.replace("Medio", "Médio")
        basin_name = basin_name.replace("Parana", "Paraná")
        
        ax = axes[i]
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
        # NOME DA BACIA NA LATERAL ESQUERDA (VERTICAL)
        ax.text(-0.1, 0.5, basin_name, 
                 transform=ax.transAxes, 
                 fontsize=24, 
                 fontweight='bold', 
                 va='center', 
                 ha='center', 
                 rotation=90)    # Rotação para ler de baixo para cima
                 #bbox=dict(facecolor='lightgray', alpha=0.2, edgecolor='none', pad=10)
            
        # Plot das séries mensais
        ax.plot(sub["YearMonth"], sub["ET_gleam"], color='black',
                markersize=4, label="ET GLEAM", linewidth=2) #marker='s'
        ax.plot(sub["YearMonth"], sub["ET"], color='red',  
                markersize=4, label="ET MOHID", linewidth=2) #marker='o'
        
        # Preenchimento entre as linhas para destacar visualmente o viés (PBIAS)
        ax.fill_between(sub["YearMonth"], sub["ET"], sub["ET_gleam"], 
                        color='gray', alpha=0.2, label="Difference (Bias)")

        # --- LINHA DE CALIBRAÇÃO / VALIDAÇÃO ---
        ax.axvline(x=split_date, color='gray', linestyle='--', linewidth=2, alpha=0.8)
        if i == 0:
            # transform=ax.get_xaxis_transform() permite usar coordenadas de dados no X e 0 a 1 no Y
            ax.text(pd.Timestamp("2013-01-01"), 0.92, "Calibration", color='black', 
                    fontsize=16, fontweight='bold', ha='center', transform=ax.get_xaxis_transform())
            ax.text(pd.Timestamp("2018-06-01"), 0.92, "Validation", color='black', 
                    fontsize=16, fontweight='bold', ha='center', transform=ax.get_xaxis_transform())
        
        ax.set_ylabel(f"ET "+ r"($\mathrm{mm \cdot yr^{-1}}$)", fontsize=20)
        ax.grid(alpha=0.3)
        
        #if i == 0:
        #    ax.set_title("Monthly ET: MOHID vs GLEAM", fontsize=20, fontweight='bold')

    plt.xlabel("Year", fontsize=20)
    
    # --- LÓGICA DA LEGENDA FORA DA FIGURA ---
    handles, labels = axes[0].get_legend_handles_labels()
    # loc='lower center' e bbox_to_anchor=(0.5, 0.0) coloca abaixo do eixo X
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=20)
    # O rect=[0, 0.05, 1, 1] garante que o tight_layout reserve espaço para a legenda no fundo
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    
    output_path = os.path.join(output_dir, "Comparison_Monthly_ET2.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Plot mensal consolidado salvo em: {output_path}")

# =================================================================
# CORRELATION PLOT
# =================================================================
def plot_mohid_correlation_analysis(df_monthly):
    # Usamos os dados mensais para entender o controle sazonal (que é o mais forte)
    basins = df_monthly["Basin"].unique()
    
    for basin in basins:
        sub = df_monthly[df_monthly["Basin"] == basin].copy()
        cols = ["ET", "Pobs", "RWC", "WTD", "WS", "TS", "T","Infiltration"] #,"RO","RO_FLOW, "Recharge""
        
        # Filtra apenas o que existe no dataframe
        available_cols = [c for c in cols if c in sub.columns]
        corr_matrix = sub[available_cols].corr(method='spearman')

        plt.figure(figsize=(10, 8))
        
        # Máscara para ver apenas o triângulo inferior (opcional, limpa o gráfico)
        #mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                    mask=None, 
                    annot=True, 
                    cmap='coolwarm', # Azul para negativo, Vermelho para positivo
                    center=0, 
                    fmt=".2f",
                    linewidths=0.5)
        
        plt.title(f"Spearman Correlation: {basin.replace('_', ' ').title()}", fontsize=20)
        plt.tight_layout()
        
        filename = f"Correlation_MOHID_{basin}.png"
        output_path = os.path.normpath(os.path.join(output_dir, filename))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Análise de correlação MOHID salva para {basin}")

# =================================================================
# ANUAL
# =================================================================
def plot_annual(df_m, df_g, output_dir):
    basins = get_sorted_basins(df_m)
    n_basins = len(basins)      
    
    label_map = {
        "ET": "ET MOHID", "ET_gleam": "ET GLEAM",
        "RWC": "RWC", "SMrz_gleam": "SMrz",
        "S_gleam": "S", "WS": "WS", "TS": "TS"
        "Pobs :" "P", "T":"T"
    }
    
    fig, axes = plt.subplots(
        nrows=n_basins, ncols=4, 
        figsize=(32, 6 * n_basins), 
        sharex='col'
    )
    if n_basins == 1: axes = np.expand_dims(axes, axis=0)
    
    plt.rcParams.update({'font.size': 20})
    border_width = 2.0
    
    # Cálculo de Limites Globais (usando o df_m como referência de anos para não distorcer)
    max_et = max(df_m["ET"].max(), df_g["ET_gleam"].max()) * 1.15
    max_s = max(df_m[["WS", "TS"]].max().max(), df_g["S_gleam"].max()) * 1.15
    max_rwc = max(df_m["RWC"].max(), df_g["SMrz_gleam"].max()) * 1.15
    # Limite para P e Infiltração (usando Pobs ou P do MOHID)
    P_col = "Pobs" if "Pobs" in df_m.columns else "P"
    max_p = max(df_m[P_col].max(), df_m["Infiltration"].max()) * 1.15
    
    for i, basin in enumerate(basins):
        # 1. Filtra MOHID
        sub_m = df_m[df_m["Basin"] == basin].sort_values("Year")
        years_m = sub_m["Year"].values
        # 2. Filtra GLEAM garantindo que só pegamos os anos que existem no MOHID
        sub_g = df_g[(df_g["Basin"] == basin) & (df_g["Year"].isin(years_m))].sort_values("Year")
        
        # 3. Sincroniza os anos (caso o GLEAM falte algum ano que o MOHID tem)
        # O ideal é usar o cruzamento exato
        years_intersect = np.intersect1d(sub_m["Year"], sub_g["Year"])
        sub_m = sub_m[sub_m["Year"].isin(years_intersect)]
        sub_g = sub_g[sub_g["Year"].isin(years_intersect)]
        
        basin_name = basin.replace("_", " ").title()
        basin_name = basin_name.replace("Medio", "Médio")
        basin_name = basin_name.replace("Parana", "Paraná")

        # ---------- COLUNA 0: ET ----------
        ax0 = axes[i, 0]
        for spine in ax0.spines.values():
            spine.set_linewidth(border_width)
        ax0.grid(alpha=0.3) 
        
        ax0.plot(years_intersect, sub_m["ET"], color='red', marker='o', label="ET MOHID")
        ax0.plot(years_intersect, sub_g["ET_gleam"], color='black', marker='s', label="ET GLEAM")
        ax0.set_ylim(0, max_et)
        if i == 0: ax0.set_title("Evapotranspiration\n"+ r"($\mathrm{mm \cdot yr^{-1}}$)", fontsize=20)
        ax0.set_ylabel(basin_name, fontsize=22, fontweight='bold')

        # ---------- COLUNA 1: STRESS ----------
        ax1 = axes[i, 1]
        for spine in ax1.spines.values():
            spine.set_linewidth(border_width)
        ax1.grid(alpha=0.3) 
        ax1.plot(years_intersect, sub_m["WS"], color='red', marker='^', label="WS")
        ax1.plot(years_intersect, sub_m["TS"], color='orange', marker='v', label="TS")
        ax1.plot(years_intersect, sub_g["S_gleam"], color='black', marker='<', label="S")
        ax1.set_ylim(0, max_s)
        if i == 0: ax1.set_title("Stress Factors\n(-)", fontsize=20)

        # ---------- COLUNA 2: RWC / SMrz ----------
        ax2 = axes[i, 2]
        for spine in ax2.spines.values():
            spine.set_linewidth(border_width)
        ax2.grid(alpha=0.3) 
        ax2.plot(years_intersect, sub_m["RWC"], color='red', marker='d', label="RWC")
        ax2.plot(years_intersect, sub_g["SMrz_gleam"], color='black', marker='x', label="SMrz")
        ax2.set_ylim(0, max_rwc)
        if i == 0: ax2.set_title("Soil Saturation\n(-)", fontsize=20)
        
        # ---------- COLUNA 3: P & INFILTRATION  ----------
        ax3 = axes[i, 3]
        for spine in ax3.spines.values():
            spine.set_linewidth(border_width)
        ax3.grid(alpha=0.3)     
        # Barra para Precipitação (fundo)
        ax3.bar(years_intersect, sub_m[P_col], color='dodgerblue', alpha=0.5, label="Precipitation")
        # Linha para Infiltração
        ax3.plot(years_intersect, sub_m["Infiltration"], color='darkblue', marker='p', linewidth=2, label="Infiltration")
        ax3.set_ylim(0, max_p)
        
        # Criando o eixo secundário para Temperatura
        ax3_t = ax3.twinx()
        for spine in ax3_t.spines.values():
            spine.set_linewidth(border_width)
            
        # Linha para Temperatura - Eixo Y Direito
        # Usamos uma cor contrastante (ex: orange ou red)
        line_t, = ax3_t.plot(years_intersect, sub_m["T"], color='tab:orange', linewidth=2.5, label="Temperature", zorder=5)
        
        # Ajuste de escala da Temperatura (ajuste os limites conforme sua região, ex: 15 a 40 graus)
        ax3_t.set_ylim(15, 40) 
        ax3_t.set_ylabel(r"T ($\mathbf{^\circ C}$)", color='tab:orange', fontsize=18, fontweight='bold')
        ax3_t.tick_params(axis='y', labelcolor='tab:orange')
        
        if i == 0: 
            ax3.set_title("Water Input & Temp\n"+ r"($\mathrm{mm \cdot yr^{-1}}$ / $^\circ \mathrm{C}$)", fontsize=20)     
        # --- APLICAÇÃO DA TIRA DE SECA (2016) ---
        for j in range(4):
            ax_target = axes[i, j]
            # Adiciona a tira cinza. alpha=0.2 para ser discreto e não tapar os dados
            ax_target.axvspan(2014.5, 2016.5, color='lightgray', alpha=0.3, label='2015-16 Drought' if i==0 and j==0 else "")

            axes[i, j].grid(alpha=0.2)
            # Define o limite do X baseado no range comum de anos
            axes[i, j].set_xlim(years_intersect.min() - 0.5, years_intersect.max() + 0.5)
            # Muda o tamanho dos números dos eixos X e Y
            axes[i, j].tick_params(axis='both', which='major', labelsize=20)
            if i == n_basins - 1: axes[i, j].set_xlabel("Year", fontsize=20)

    # Legenda e salvamento (mesma lógica anterior)
    handles_all, labels_all = [], []
    for j in range(4):
            ax_leg = axes[0, j]
            h, l = ax_leg.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li and li not in labels_all:
                    handles_all.append(hi)
                    labels_all.append(li)
                    
    fig.legend(handles_all, labels_all, loc='lower center', ncol=10, 
               bbox_to_anchor=(0.5, 0.02), fontsize=20)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    filename = "Anual2.png"
    output_path = os.path.normpath(os.path.join(output_dir, filename))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Atlas anual sincronizado salvo em: {output_path}")

# =================================================================
# CLIMATOLOGIA
# =================================================================
def plot_climatology(df_climat_m, df_climat_g):
    basins = get_sorted_basins(df_climat_m)
    n_basins = len(basins)
    
     
    # --- 1. CÁLCULO DE LIMITES GLOBAIS (Para escalas iguais) ---
    # Coluna 1
    max_et = max(df_climat_m["ET"].max(), df_climat_g["ET_gleam"].max()) * 1.1
    max_p = df_climat_m["Pobs"].max() * 1.16
    
    # Coluna 2
    max_recharge = df_climat_m["Infiltration"].max() * 1.1
    max_wtd = df_climat_m["WTD"].max() * 1.1
    min_wtd = df_climat_m["WTD"].min() * 0.9 # Para não colar no topo ao inverter
    
    # Criamos uma matriz de n_basins linhas e 2 colunas
    fig, axes = plt.subplots(nrows=n_basins, ncols=2, figsize=(20, 5 * n_basins), sharex='col')
    plt.rcParams.update({'font.size': 20})
    border_width = 2.0
    
    # Garantir que axes seja 2D mesmo com 1 bacia
    if n_basins == 1: axes = np.expand_dims(axes, axis=0)

    for i, basin in enumerate(basins):
        # --- PREPARAÇÃO DE DADOS ---
        col_mes = "Month" if "Month" in df_climat_m.columns else "month"
        m_sub = df_climat_m[df_climat_m["Basin"] == basin].sort_values(col_mes)
        g_sub = df_climat_g[df_climat_g["Basin"] == basin].sort_values(col_mes)
        months = m_sub[col_mes]
        basin_name = basin.replace('_', ' ').title()
        basin_name = basin_name.replace("Medio", "Médio")
        basin_name = basin_name.replace("Parana", "Paraná")
        
        
        # ==========================================================
        # COLUNA 1: ET, CHUVA E STRESS (O que você tinha na função 1)
        # ==========================================================
        ax1 = axes[i, 0]
        for spine in ax1.spines.values():
            spine.set_linewidth(border_width)   
        ax1.grid(alpha=0.3)    

        # NOME DA BACIA NA LATERAL ESQUERDA (VERTICAL)
        ax1.text(-0.18, 0.5, basin_name, 
                 transform=ax1.transAxes, 
                 fontsize=24, 
                 fontweight='bold', 
                 va='center', 
                 ha='center', 
                 rotation=90)    # Rotação para ler de baixo para cima
                 #bbox=dict(facecolor='lightgray', alpha=0.2, edgecolor='none', pad=10)
        # ET
        ln1 = ax1.plot(months, m_sub["ET"], color='red', marker='o', label="ET MOHID", linewidth=2)
        ln2 = ax1.plot(months, g_sub["ET_gleam"], color='black', marker='s', label="ET GLEAM")
        ax1.set_ylabel("ET " + r"($\mathrm{mm \cdot month^{-1}}$)",  fontsize=20)
        #ax1.set_ylabel(rf"{basin_name}" + "\n" + r"ET ($\mathbf{mm \cdot month^{-1}}$)", fontweight='bold', fontsize=20)
        ax1.set_ylim(0, max_et) # LIMITE FIXO
                
        # Precipitação (Eixo 2)
        ax1_p = ax1.twinx()
        bar1 = ax1_p.bar(months, m_sub["Pobs"], color='dodgerblue', alpha=0.5, label="P")
        ax1_p.set_ylabel("P " + r"($\mathrm{mm \cdot month^{-1}}$)", color='dodgerblue', fontsize=20)
        ax1_p.tick_params(axis='y', labelcolor='dodgerblue', labelsize=16)
        ax1_p.set_ylim(0, max_p) # LIMITE FIXO
        
        # Stress (Eixo 3 - Deslocado)
        ax1_s = ax1.twinx()
        ax1_s.spines['right'].set_position(('outward', 70))
        ln3 = ax1_s.plot(months, m_sub["WS"], color='darkgreen', linestyle='-', linewidth=2, label="WS (MOHID)")
        if "S_gleam" in g_sub.columns:
            ln4 = ax1_s.plot(months, g_sub["S_gleam"], color='darkgreen', linestyle='--', linewidth=2, label="Stress (GLEAM)")
        ax1_s.tick_params(axis='y', labelcolor='darkgreen', labelsize=16)
        ax1_s.set_ylabel("Stress Factor (-)", color='darkgreen', fontsize=20)
        ax1_s.set_ylim(0, 1.1)

        #ax1.set_title(f"{basin_name}: P, ET & Stress", loc='left', fontsize=26, fontweight='bold')

        # ==========================================================
        # COLUNA 2: RECARGA, WTD E RWC (O que você tinha na função 2)
        # ==========================================================
        ax2 = axes[i, 1]
        for spine in ax2.spines.values():
            spine.set_linewidth(border_width)
        ax2.grid(alpha=0.3) 
        
        # 1. WTD (Lençol Freático) agora é o eixo principal (esquerda)
        ax2.plot(months, m_sub["WTD"], color='red', marker='o', linewidth=2, label="WTD")
        ax2.set_ylabel("Water Table Depth (m)", color='black', fontsize=20)
        ax2.set_ylim(0, max_wtd)
        ax2.invert_yaxis() # Mantém a inversão (água subindo = linha subindo)
        
        # 2. Recarga agora é o primeiro twinx (direita interna)
        ax2_w = ax2.twinx()
        ax2_w.bar(months, m_sub["Infiltration"], color='dodgerblue', alpha=0.5, label="Infiltration")
        ax2_w.set_ylabel("Infiltration " + r"($\mathrm{mm \cdot month^{-1}}$)", color='dodgerblue', fontsize=20)
        ax2_w.tick_params(axis='y', labelcolor='dodgerblue', labelsize=16)
        ax2_w.set_ylim(0, max_recharge) # LIMITE FIXO GLOBAL
        
        # Umidade Solo (Eixo 3 - Deslocado)
        ax2_r = ax2.twinx()
        ax2_r.spines['right'].set_position(('outward', 70))
        ax2_r.plot(months, m_sub["RWC"], color='darkgreen', label="RWC")
        if "SMrz_gleam" in g_sub.columns:
            ax2_r.plot(months, g_sub["SMrz_gleam"], color='darkgreen', linestyle='--', linewidth=2, label="SMrz (GLEAM)")
        ax2_r.tick_params(axis='y', labelcolor='darkgreen', labelsize=16)
        ax2_r.set_ylabel("Porous Media Saturation (-)", color='darkgreen', fontsize=20)
        ax2_r.set_ylim(0, 1)

        #ax2.set_title(f"{basin_name}: Porous Media dynamic", loc='left', fontsize=26, fontweight='bold')

        # Estética Geral para cada linha
        for ax in [ax1, ax2]:
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            ax.grid(axis='y', alpha=0.15)

    plt.tight_layout()

    # Coleta de legendas (apenas uma vez)
    handles, labels = [], []
    # Usamos o último subplot para pegar as legendas de todos os eixos possíveis
    for axis_obj in [ax1, ax1_p, ax1_s, ax2, ax2_w, ax2_r]:
        h, l = axis_obj.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=20)
    
    output_path = os.path.join(output_dir, "Climatologia.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Atlas consolidado salvo em: {output_path}")

output_dir = "./outputs" # Certifique-se que esta pasta existe
if not os.path.exists(output_dir): os.makedirs(output_dir)
    
# --- MOHID ---
path1 = r"F:\Debora\Water Balance\ET\Tocantins\basin_outputs"
path2 = r"F:\Debora\Water Balance\ET\Tocantins\basin_outputs2"
files1 = glob(os.path.join(path1, "*_daily.csv"))
files2 = glob(os.path.join(path2, "*_daily.csv"))
df_mohid1 = load_mohid_files(files1, "1")
df_mohid2 = load_mohid_files(files2, "2")
df_mohid = pd.concat([df_mohid1, df_mohid2], ignore_index=True)
df_mohid = df_mohid.groupby(["Date", "Basin"]).first().reset_index()

df_mohid["Date"] = pd.to_datetime(df_mohid["Date"]).dt.normalize()
df_mohid["Basin"] = df_mohid["Basin"].str.strip().str.lower()
df_mohid = df_mohid.rename(columns={"PM": "WTD"})

accumulate_vars = ["P","Pobs","ET","RO", "Infiltration"]
mean_vars = ["WTD", "RWC", "RO_FLOW", "T", "WS", "TS"] # Trocado PM por WTD

# 1. Agregação Mensal (Base para o cálculo correto)
df_monthly_mohid, _, _ = aggregate_timescales(
    df_mohid,
    accumulate_vars,
    mean_vars,
    date_col="Date"
)

P_col = "Pobs" if "Pobs" in df_monthly_mohid.columns else "P"
# Recarga Mensal = O que sobrou da chuva após ET e RO
df_monthly_mohid['Recharge'] = (df_monthly_mohid[P_col] - 
                                df_monthly_mohid['ET'] - 
                                df_monthly_mohid['RO']).clip(lower=0)
# Deficiência Mensal = Quando a ET é maior que a Chuva
df_monthly_mohid['Deficiency'] = (df_monthly_mohid['ET'] - 
                                  df_monthly_mohid[P_col]).clip(lower=0)
# Razão ET/P (Índice de Aridez local)
df_monthly_mohid['ETP_ratio'] = df_monthly_mohid['ET'] / df_monthly_mohid[P_col].replace(0, np.nan)

_, df_annual_mohid, df_climat_mohid = aggregate_timescales(
    df_monthly_mohid, 
    accumulate_vars, 
    mean_vars, 
    date_col="YearMonth" 
)

# --- GLEAM ---
path_gleam = r"F:\Debora\Water Balance\ET\Gleam"
files = [
    os.path.join(path_gleam, "Consolidado_Diario_ET.csv"),
    os.path.join(path_gleam, "Consolidado_Diario_S.csv"),
    os.path.join(path_gleam, "Consolidado_Diario_SMrz.csv")
]
var_names = ["ET_gleam", "S_gleam", "SMrz_gleam"]
df_gleam = load_gleam(files, var_names)
df_gleam["Date"] = pd.to_datetime(df_gleam["Date"]).dt.normalize()
df_gleam["Basin"] = df_gleam["Basin"].str.strip().str.lower()

accumulate_vars = ["ET_gleam"]
mean_vars = ["S_gleam", "SMrz_gleam"]
df_monthly_gleam, df_annual_gleam, df_climat_gleam = aggregate_timescales(
    df_gleam,
    accumulate_vars,
    mean_vars,
    date_col="Date"
)

# --- ALL ANNUAL DFS ---
df_total = df_annual_mohid.merge(
    df_annual_gleam,
    on=["Year","Basin"],
    how="inner"
)

# Exportar Climatologia MOHID
df_climat_mohid.to_csv(os.path.join(output_dir, "Climatologia_Mensal_MOHID.csv"), index=False, sep=';')
# Exportar Climatologia GLEAM
df_climat_gleam.to_csv(os.path.join(output_dir, "Climatologia_Mensal_GLEAM.csv"), index=False, sep=';')

print("Arquivos de climatologia exportados com sucesso para a pasta de outputs.")
# --- EXECUÇÃO ---
# 1. Stats Diários
df_daily_stats = calculate_all_stats(df_mohid, df_gleam, "Date", "Date")
df_daily_stats.to_csv(os.path.join(output_dir, "Stats_ET_Diario_CalVal.csv"), index=False, sep=';')

# 2. Stats Mensais
df_monthly_stats = calculate_all_stats(df_monthly_mohid, df_monthly_gleam, "YearMonth", "YearMonth")
df_monthly_stats.to_csv(os.path.join(output_dir, "Stats_ET_Mensal_CalVal.csv"), index=False, sep=';')

print("✅ Estatísticas de Calibração e Validação calculadas!")
# Executar
plot_ET_monthly(df_monthly_mohid, df_monthly_gleam, output_dir)
# Executar apenas com o dataframe mensal do MOHID
plot_mohid_correlation_analysis(df_monthly_mohid)
# Executar
plot_annual(df_annual_mohid, df_annual_gleam, output_dir)

# 1. Selecionar apenas as colunas de interesse
cols_interesse = [
    "Basin", "Year", 
    "ET", "ET_gleam",           # Evapotranspiração
    "RWC", "SMrz_gleam", 'WTD',       # Umidade Solo
    "WS", "TS", "S_gleam",       # Stresses (Water, Temperature e GLEAM Stress)
    "Pobs",  "T", "Infiltration"
]

# 2. Criar o DataFrame final filtrado
df_tabela_anual = df_total[cols_interesse].copy()
# 3. Organizar por Bacia e Ano para facilitar a leitura
df_tabela_anual = df_tabela_anual.sort_values(["Basin", "Year"])
# 4. Arredondar os valores para 3 casas decimais (opcional, para limpeza)
df_tabela_anual = df_tabela_anual.round(3)
# 5. Salvar em CSV
df_tabela_anual.to_csv(os.path.join(output_dir, "Tabela_Anual_MOHID_GLEAM.csv"), index=False, sep=';')
print("Tabela consolidada gerada com sucesso:")

# Executar
plot_climatology(df_climat_mohid, df_climat_gleam)
