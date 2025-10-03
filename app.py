# app.py — Tereos Risk Dashboard (Streamlit)
# ------------------------------------------
# Ajouts :
# - Config PAR POSTE en tête de sidebar (limit + seuils + on/off usage), persistée en session
# - Suppression des sliders globaux d’alerte/critique
# - Info-bulles exhaustives + onglet Méthodologie
# - Fusion multi-templates (last wins) et auto-refresh (cache TTL) inchangés

import os, re, glob, io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# --------------------- CONFIG & BRANDING ---------------------
st.set_page_config(page_title="Tereos | Risk Dashboard", page_icon="", layout="wide")

TEREOS_PRIMARY = "#8c1d13"
TEREOS_TEXT = "#0b0d12"

st.markdown(f"""
<style>
  html, body, [class*="css"] {{
    font-family:'Inter',sans-serif;
    color:{TEREOS_TEXT};
  }}
  .stApp {{
    background:linear-gradient(180deg,#ffffff 0%,#f5f7fa 100%);
  }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  .ter-card { background:#ffffff; padding:16px 18px; border:1px solid rgba(15,23,42,.06);
              border-radius:16px; box-shadow:0 6px 24px rgba(15,23,42,.08); }

  .header-wrap { display:flex; align-items:center; gap:28px; }
  .title-block { min-width:340px; }
  .kpi-wrap { display:grid; grid-template-columns:repeat(4, minmax(150px,1fr)); gap:18px; align-items:stretch; flex:1; }
  .badge-wrap { display:flex; align-items:center; justify-content:flex-end; min-width:240px; }

  .kpi{ display:flex; flex-direction:column; justify-content:center; gap:6px; padding:12px 16px;
        background:#fff; border:1px solid rgba(15,23,42,.08); border-radius:14px; min-height:110px; }
  .kpi .label{ font-size:12px; opacity:.7; display:flex; align-items:center; gap:6px; }
  .kpi .value{ font-size:28px; font-weight:800; line-height:1.1; color:#0b0d12; }
  .kpi .sub{ font-size:12px; opacity:.6; }

  .title-accent{ font-weight:800; letter-spacing:.3px; font-size:24px; margin-bottom:6px;
                 background:linear-gradient(90deg,#8c1d13,#f0a500);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

  .risk-badge{ display:inline-flex; align-items:center; gap:10px; padding:12px 18px; border-radius:999px;
               font-weight:800; letter-spacing:.3px; border:1px solid rgba(15,23,42,.08); background:rgba(255,159,10,.18); }
  .risk-dot{ width:10px; height:10px; border-radius:50%; display:inline-block; background:#ff9f0a; }

  .hint{ cursor:help; display:inline-flex; align-items:center; justify-content:center;
         border:1px dashed rgba(15,23,42,.25); border-radius:50%;
         width:16px; height:16px; font-size:11px; color:#0b0d12; background:#fff; }
</style>
""", unsafe_allow_html=True)

# --------------------- TOOLTIPS ---------------------
EXPLAIN = {
    # KPI & Global
    "VaR (kUSD)": "Value at Risk : perte potentielle max attendue (ex. 1 jour, 95%). Unités : milliers de USD.",
    "Usage moyen": "Moyenne des ratios Value/Limit (postes avec Usage actif).",
    "Usage pic": "Plus haut ratio Value/Limit observé parmi les postes actifs.",
    "Alertes actives": "Nombre de postes en ‘Surveillance’ (≥ alerte) ou ‘Critique’ (≥ critique).",
    "Value": "Valeur exposée/mesurée pour le poste (unité selon le poste).",
    "Delta": "Variation vs période précédente dans les données source.",
    "Limit": "Limite de risque allouée au poste.",
    "Usage": "Rapport Value/Limit (sans %).",
    "Usage_sens": "Position / Risk limits (sans %).",

    # Marchés & métriques
    "Flat price": "Prix spot/comptant du sucre physique (hors base/transport).",
    "NY spread": "Écart de prix lié au marché NY (ex. différence entre contrats ou vs spot).",
    "LDN spread": "Écart de prix lié au marché Londres (similaire au NY spread, côté LDN).",
    "White premium": "Prime du sucre blanc vs sucre brut (proxy marge raffinage).",
    "Futures": "Exposition sur contrats à terme par échéance.",
    "LDN": "Séries liées au marché de Londres (ICE Europe), sucre blanc.",
    "NY": "Séries liées au marché de New York (ICE US), sucre brut.",
    "Option Delta": "Sensibilité Δ : variation de l’option pour 1 pt de sous-jacent.",
    "CSO (lots)": "Calendar Spread Options — taille en lots.",
    "Gamma (lots/pt)": "Variation du delta pour 1 pt de sous-jacent (lots/pt).",
    "Vega ($/vol.pt)": "Variation de valeur pour +1 pt de volatilité (USD).",

    # Tableaux
    "Parameters": "Nom du paramètre de sensibilité (Delta, Gamma, Vega, etc.).",
    "Position": "Taille de position associée au paramètre (unités indiquées).",
    "Risk limits": "Limite de risque (même unité que Position).",

    # Divers UI
    "alpha": "Paramètre de lissage exponentiel (EWMA). Plus élevé = plus réactif.",
    "horizon": "Nombre d’échéances projetées par la prévision EWMA.",
    "metrics_sel": "Choisissez les séries à afficher dans la vue Échéances."
}

# --------------------- HELPERS (détection/fusion) ---------------------
def _scan_templates():
    base_dirs = [r"C:\Users\azad1\Desktop\Tereos\Ismail", os.getcwd(), "/mnt/data"]
    paths = []
    for bd in base_dirs:
        if os.path.isdir(bd):
            paths += glob.glob(os.path.join(bd, "template*.xlsx"))
    def num(p):
        m = re.search(r"template(\d+)\.xlsx$", os.path.basename(p), re.I)
        return int(m.group(1)) if m else -1
    paths = [p for p in paths if num(p) >= 0]
    paths.sort(key=lambda p: num(p))
    return paths

@st.cache_data(ttl=5, show_spinner=False)
def _read_all_templates(paths):
    out = []
    for p in paths:
        try:
            mtime = os.path.getmtime(p)
            xls = pd.read_excel(p, sheet_name=None, header=None, engine="openpyxl")
            out.append((p, mtime, xls))
        except Exception:
            continue
    return out

def normalize_headers(df: pd.DataFrame):
    df2 = df.copy()
    if df2.shape[0] > 0 and df2.iloc[0].notna().sum() >= 2:
        df2.columns = df2.iloc[0].astype(str).str.strip()
        df2 = df2[1:].reset_index(drop=True)
    else:
        df2.columns = [f"col_{i}" for i in range(df2.shape[1])]
    return df2

def is_global_table(df):
    cols = [c.lower() for c in df.columns.astype(str)]
    needed = ["value", "delta", "limit", "usage"]
    first_col_like = any("var" in str(x).lower() or "flat" in str(x).lower() for x in df.iloc[:,0].astype(str))
    return all(any(n in c for c in cols) for n in needed) and first_col_like

def is_maturities_table(df):
    labels = ["flat price","ldn","ny","futures","option delta","white premium","ny spread","ldn spread"]
    first_col = df.columns[0].lower()
    has_labels = any(l in " ".join(df[first_col].astype(str).str.lower().tolist()) for l in labels)
    other_cols = df.columns[1:]
    looks_dates = sum([("-" in str(c) or "/" in str(c) or "25" in str(c) or "26" in str(c)) for c in other_cols]) >= 2
    return has_labels and looks_dates

def is_sensitivities_table(df):
    cols = [c.lower() for c in df.columns.astype(str)]
    needed = ["parameters", "position", "risk", "usage"]
    return all(any(n in c for c in cols) for n in needed)

def coerce_numeric(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([np.nan]*len(s))

def _overwrite_frame(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy()
    cols = list(dict.fromkeys(list(left.columns) + list(right.columns)))
    l = left.reindex(columns=cols)
    r = right.reindex(columns=cols)
    cat = pd.concat([l, r], ignore_index=True)
    cat = cat.dropna(subset=[key], how="all")
    cat[key] = cat[key].astype(str)
    cat = cat.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    return cat

def _overwrite_wide_rows(left: pd.DataFrame, right: pd.DataFrame, row_key: str) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy()
    all_cols = list(dict.fromkeys(list(left.columns) + list(right.columns)))
    l = left.reindex(columns=all_cols)
    r = right.reindex(columns=all_cols)
    comb = pd.concat([l, r], ignore_index=True)
    def agg_last(series):
        for v in reversed(series.tolist()):
            if pd.notna(v): return v
        return np.nan
    grouped = comb.groupby(row_key, dropna=False).agg(agg_last).reset_index()
    cols = [row_key] + [c for c in grouped.columns if c != row_key]
    return grouped[cols]

# --------------------- LECTURE + FUSION ---------------------
paths = _scan_templates()
if not paths:
    st.error("Aucun fichier `templateX.xlsx` trouvé. Placez template1.xlsx (ou supérieur) dans un dossier scanné.")
    st.stop()

all_files = _read_all_templates(paths)  # TTL=5s → auto-refresh
global_acc = matur_acc = sens_acc = None
active_path, active_mtime = paths[-1], os.path.getmtime(paths[-1])

for path in paths:
    sheets = dict()
    for p, m, xls in all_files:
        if p == path:
            sheets = xls; break
    if not sheets: continue

    g_df = m_df = s_df = None
    for name, raw in sheets.items():
        df = normalize_headers(raw).dropna(how="all").dropna(axis=1, how="all")
        if df.empty or df.shape[1] < 2: continue
        if g_df is None and is_global_table(df): g_df = df.copy()
        elif m_df is None and is_maturities_table(df): m_df = df.copy()
        elif s_df is None and is_sensitivities_table(df): s_df = df.copy()

    if any(x is None for x in [g_df, m_df, s_df]):
        for name, raw in sheets.items():
            df = raw.copy()
            empties = df.isna().all(axis=1)
            blocks, start = [], None
            for i, is_empty in empties.items():
                if not is_empty and start is None: start = i
                if (is_empty or i == df.index.max()) and start is not None:
                    end = i if is_empty else i+1
                    blk = df.iloc[start:end].reset_index(drop=True)
                    blk = normalize_headers(blk).dropna(how="all").dropna(axis=1, how="all")
                    if blk.shape[1] >= 2: blocks.append(blk)
                    start = None
            for blk in blocks:
                if g_df is None and is_global_table(blk): g_df = blk
                elif m_df is None and is_maturities_table(blk): m_df = blk
                elif s_df is None and is_sensitivities_table(blk): s_df = blk

    if g_df is not None:
        g_df.rename(columns=lambda x: str(x).strip(), inplace=True)
        if "Value" not in g_df.columns:
            g_df.columns = [g_df.columns[0], "Value", "Delta", "Limit", "Usage"][:len(g_df.columns)]
        g_df.rename(columns={g_df.columns[0]:"Poste"}, inplace=True)
        for c in ["Value","Delta","Limit","Usage"]:
            if c in g_df.columns: g_df[c] = coerce_numeric(g_df[c])
        if "Usage" not in g_df.columns or g_df["Usage"].isna().all():
            g_df["Usage"] = g_df.apply(
                lambda r: (float(r["Value"])/float(r["Limit"]))
                if pd.notna(r.get("Limit")) and r.get("Limit") not in (0, "0") else np.nan, axis=1
            )
        global_acc = _overwrite_frame(global_acc, g_df, key="Poste")

    if m_df is not None:
        m_df.rename(columns=lambda x: str(x).strip(), inplace=True)
        m_df.rename(columns={m_df.columns[0]:"Metric"}, inplace=True)
        for c in [c for c in m_df.columns if c!="Metric"]:
            m_df[c] = coerce_numeric(m_df[c])
        matur_acc = _overwrite_wide_rows(matur_acc, m_df, row_key="Metric")

    if s_df is not None:
        s_df.rename(columns=lambda x: str(x).strip(), inplace=True)
        mapping = {}
        for c in s_df.columns:
            cl = c.lower()
            if "param" in cl: mapping[c] = "Parameters"
            elif "position" in cl: mapping[c] = "Position"
            elif "risk" in cl: mapping[c] = "Risk limits"
            elif "usage" in cl: mapping[c] = "Usage"
            elif "gamma" in cl: mapping[c] = "Gamma (lots/pt)"
            elif "vega" in cl: mapping[c] = "Vega ($/vol.pt)"
            elif "delta" in cl: mapping[c] = "Option Delta"
            elif "cso"   in cl: mapping[c] = "CSO (lots)"
        s_df.rename(columns=mapping, inplace=True)
        for c in ["Position","Risk limits","Usage"]:
            if c in s_df.columns: s_df[c] = coerce_numeric(c=s_df[c]) if False else coerce_numeric(s_df[c])
        sens_acc = _overwrite_frame(sens_acc, s_df, key="Parameters")

if any(x is None for x in [global_acc, matur_acc, sens_acc]):
    st.error("Impossible d’identifier et fusionner les 3 tableaux à partir des templates. Vérifie les entêtes.")
    st.stop()

m_long = matur_acc.melt(id_vars=["Metric"], var_name="Echeance", value_name="Valeur")
m_long["Valeur"] = coerce_numeric(m_long["Valeur"])

# --------------------- SIDEBAR (CONFIG PAR POSTE EN HAUT) ---------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/9/98/Logo_Tereos_2016.png",
    caption="Tereos – Risk & Markets",
    use_container_width=True
)

# Méta : valeurs par défaut si un poste n’a pas de réglage dédié
WARN_DEFAULT = 0.80
CRIT_DEFAULT = 1.00

st.sidebar.markdown("### Configurer un poste")
if "POST_OVERRIDES" not in st.session_state:
    st.session_state.POST_OVERRIDES = {}  # {poste_lower: {"limit":float,"warn":float,"crit":float,"apply":bool}}

postes = list(global_acc["Poste"].astype(str).dropna().unique())
poste_sel = st.sidebar.selectbox("Poste", postes, help="Choisissez le poste à paramétrer.")

k = str(poste_sel).strip().lower()
ov = st.session_state.POST_OVERRIDES.get(k, {})
file_limit = global_acc.loc[global_acc["Poste"]==poste_sel, "Limit"].dropna().head(1)
limit_default = float(ov.get("limit", file_limit.iloc[0] if len(file_limit) else 0.0))
apply_default = bool(ov.get("apply", True))
warn_default  = float(ov.get("warn", WARN_DEFAULT))
crit_default  = float(ov.get("crit", CRIT_DEFAULT))

with st.sidebar.popover("ℹ️ Définition", use_container_width=True):
    st.markdown(f"**{poste_sel}** — {EXPLAIN.get(poste_sel,'Définition indisponible.')}")
    st.caption("Value = exposition ; Limit = plafond ; Usage = Value / Limit.")

limit_input = st.sidebar.number_input("Limit du poste", min_value=0.0, step=1.0, value=float(limit_default), help=EXPLAIN["Limit"])
apply_input = st.sidebar.toggle("Calculer Usage (%) pour ce poste ?", value=apply_default, help="OFF = pas d’Usage ni de Status pour ce poste.")
c1, c2 = st.sidebar.columns(2)
with c1:
    warn_input = st.number_input("Alerte (ratio)", min_value=0.0, step=0.05, value=float(warn_default))
with c2:
    crit_input = st.number_input("Critique (ratio)", min_value=0.0, step=0.05, value=float(crit_default))

if st.sidebar.button("Enregistrer", use_container_width=True):
    st.session_state.POST_OVERRIDES[k] = {
        "limit": float(limit_input),
        "apply": bool(apply_input),
        "warn": float(warn_input),
        "crit": float(crit_input),
    }
    st.sidebar.success("Réglage poste enregistré.")

if st.session_state.POST_OVERRIDES:
    st.sidebar.markdown("#### Réglages enregistrés")
    oview = pd.DataFrame([{"Poste": n.title(), **v} for n, v in st.session_state.POST_OVERRIDES.items()])
    st.sidebar.dataframe(oview, use_container_width=True, hide_index=True)

st.sidebar.divider()
alpha_ewm = st.sidebar.slider("Alpha EWMA (prévision)", 0.1, 0.9, 0.5, 0.1, help=EXPLAIN["alpha"])
forecast_steps = st.sidebar.select_slider("Horizon de prévision (échéances)", options=[1,2,3,4,5,6], value=3, help=EXPLAIN["horizon"])

st.sidebar.divider()
st.sidebar.markdown(
    f"**Fichier le plus récent** : `{os.path.basename(active_path)}`  \n"
    f"*Modifié le* : {datetime.fromtimestamp(active_mtime):%d %b %Y %H:%M:%S}  \n"
    f"_Fusion de {len(paths)} template(s) (priorité au dernier)._"
)
st.sidebar.markdown("### Filtres – Maturities")
metrics_sel = st.sidebar.multiselect(
    "Metrics affichées",
    options=["Flat price", "White premium", "NY spread", "LDN spread", "LDN", "NY", "Futures", "Option Delta"],
    default=["Flat price","LDN","NY","Futures"],
    help=EXPLAIN["metrics_sel"]
)

# --------------------- LOGIQUE POSTE (seuils/limit par ligne) ---------------------
def status_from_usage(u, warn, crit):
    if pd.isna(u): return "N/A"
    if u >= crit:  return "🔴 Critique"
    if u >= warn:  return "🟠 Surv."
    return "🟢 OK"

def apply_overrides(df: pd.DataFrame):
    out = df.copy()
    out["Limit_applied"] = out["Limit"]
    out["Warn_applied"]  = WARN_DEFAULT
    out["Crit_applied"]  = CRIT_DEFAULT
    out["ApplyUsage"]    = True

    for idx, r in out.iterrows():
        key = str(r["Poste"]).strip().lower()
        cfg = st.session_state.POST_OVERRIDES.get(key)
        if cfg:
            out.at[idx, "Limit_applied"] = cfg.get("limit", r["Limit"])
            out.at[idx, "Warn_applied"]  = cfg.get("warn",  WARN_DEFAULT)
            out.at[idx, "Crit_applied"]  = cfg.get("crit",  CRIT_DEFAULT)
            out.at[idx, "ApplyUsage"]    = bool(cfg.get("apply", True))

        # compute Usage with applied settings
        if (not out.at[idx, "ApplyUsage"]) or pd.isna(out.at[idx, "Limit_applied"]) or out.at[idx, "Limit_applied"] in (0, "0") or pd.isna(out.at[idx, "Value"]):
            out.at[idx, "Usage"] = np.nan
        else:
            try:
                out.at[idx, "Usage"] = float(out.at[idx, "Value"]) / float(out.at[idx, "Limit_applied"])
            except Exception:
                out.at[idx, "Usage"] = np.nan

    out["Usage %"] = (out["Usage"]*100).round(1)
    out["Status"]  = out.apply(lambda r: status_from_usage(r["Usage"], r["Warn_applied"], r["Crit_applied"]), axis=1)
    return out

# --------------------- HEADER & KPIs ---------------------
st.markdown("<div class='header-wrap'>", unsafe_allow_html=True)
st.markdown(f"""
  <div class='title-block'>
    <div class='title-accent'>Tereos – Global Risk Dashboard</div>
    <div style="font-size:13px;opacity:.75;">
      <b>Sources fusionnées :</b><br/>
      <span style="color:#0ea5e9;">{'; '.join(os.path.basename(p) for p in paths)}</span><br/>
      {datetime.now():%d %b %Y %H:%M}
    </div>
  </div>
""", unsafe_allow_html=True)

g_applied = apply_overrides(global_acc)

var_val    = float(g_applied.loc[g_applied["Poste"].str.lower().str.contains("var"), "Value"].fillna(0).sum())
usage_mean = float(g_applied["Usage"].mean(skipna=True)) if g_applied["Usage"].notna().any() else np.nan
usage_max  = float(g_applied["Usage"].max(skipna=True)) if g_applied["Usage"].notna().any() else np.nan
n_alerts   = int(((g_applied["Usage"]>=g_applied["Warn_applied"]) & (g_applied["Usage"]<g_applied["Crit_applied"])).sum()
                 + (g_applied["Usage"]>=g_applied["Crit_applied"]).sum()) if g_applied["Usage"].notna().any() else 0

if pd.isna(usage_max):
    badge_bg, dot, label = "rgba(107,114,128,.18)", "#6b7280", "NIVEAU INCONNU"
elif usage_max >= CRIT_DEFAULT:
    badge_bg, dot, label = "rgba(255,59,48,.18)", "#ff3b30", "RISQUE ÉLEVÉ"
elif usage_max >= WARN_DEFAULT:
    badge_bg, dot, label = "rgba(255,159,10,.18)", "#ff9f0a", "RISQUE MODÉRÉ"
else:
    badge_bg, dot, label = "rgba(48,209,88,.18)", "#30d158", "RISQUE FAIBLE"

st.markdown(f"""
  <div class='kpi-wrap'>
    <div class='kpi'>
      <div class='label'>VaR (kUSD) <span class='hint' title="{EXPLAIN['VaR (kUSD)']}">i</span></div>
      <div class='value'>{var_val:,.0f}</div>
      <div class='sub'>Horizon 1J (ex.)</div>
    </div>
    <div class='kpi'>
      <div class='label'>Usage moyen <span class='hint' title="{EXPLAIN['Usage moyen']}">i</span></div>
      <div class='value'>{(usage_mean*100 if not np.isnan(usage_mean) else 0):,.1f}%</div>
      <div class='sub'>Postes actifs</div>
    </div>
    <div class='kpi'>
      <div class='label'>Usage pic <span class='hint' title="{EXPLAIN['Usage pic']}">i</span></div>
      <div class='value'>{(usage_max*100 if not np.isnan(usage_max) else 0):,.1f}%</div>
      <div class='sub'>vs limites</div>
    </div>
    <div class='kpi'>
      <div class='label'>Alertes actives <span class='hint' title="{EXPLAIN['Alertes actives']}">i</span></div>
      <div class='value'>{n_alerts}</div>
      <div class='sub'>Seuils par poste</div>
    </div>
  </div>
""", unsafe_allow_html=True)

st.markdown(f"""
  <div class='badge-wrap'>
    <div class='risk-badge' style='background:{badge_bg}'>
      <span class='risk-dot' style='background:{dot}'></span> {label}
    </div>
  </div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------- TABS ---------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🧮 Maturities & Forecast", "🚨 Alerting", "🧰 Sensitivities", "📘 Méthodologie"])

# ==== TAB 1: DASHBOARD GLOBAL ====
with tab1:
    st.markdown("#### Risques globaux")

    colcfg = {
        "Poste": st.column_config.TextColumn("Poste", help="Nom du poste (ex. VaR, Flat price, NY/LDN spread, White premium…)."),
        "Value": st.column_config.NumberColumn("Value", help=EXPLAIN["Value"], format="%.2f"),
        "Delta": st.column_config.NumberColumn("Delta", help=EXPLAIN["Delta"], format="%.2f"),
        "Limit_applied": st.column_config.NumberColumn("Limit (appliquée)", help="Limite utilisée (override poste si défini).", format="%.2f"),
        "Usage %": st.column_config.NumberColumn("Usage (%)", help=EXPLAIN["Usage"], format="%.1f"),
        "Status": st.column_config.TextColumn("Status", help="Calculé avec les seuils du poste."),
        "ApplyUsage": st.column_config.CheckboxColumn("Apply", help="ON = Usage calculé pour ce poste."),
        "Warn_applied": st.column_config.NumberColumn("Alerte (poste)", help="Seuil d'alerte appliqué.", format="%.2f"),
        "Crit_applied": st.column_config.NumberColumn("Critique (poste)", help="Seuil critique appliqué.", format="%.2f"),
    }
    to_display = g_applied[["Poste","Value","Delta","Limit_applied","Usage %","Status","ApplyUsage","Warn_applied","Crit_applied"]]
    st.dataframe(to_display, use_container_width=True, hide_index=True, column_config=colcfg)

    chart_df = g_applied[g_applied["ApplyUsage"] & g_applied["Usage"].notna()].copy()
    fig = px.bar(
        chart_df, x="Poste", y="Usage %",
        title="Usage par poste (%)",
        text="Usage %",
        hover_data={"Value":":.2f","Limit_applied":":.2f","Usage":":.2%","Warn_applied":":.2f","Crit_applied":":.2f"}
    )
    fig.update_traces(
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Usage: %{y:.1f}%<br>Valeur: %{customdata[0]:.2f}<br>Limite: %{customdata[1]:.2f}<br>Ratio: %{customdata[2]:.2%}<br>Alerte: %{customdata[3]:.2f} | Critique: %{customdata[4]:.2f}<extra></extra>"
    )
    # plus de lignes globales (seuils par poste visibles dans le hover/table)
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ==== TAB 2: MATURITIES & FORECAST ====
def ewm_forecast(series, steps=3, alpha=0.5):
    s = pd.Series(series).dropna()
    if s.empty: return [np.nan]*steps
    ew = s.ewm(alpha=alpha).mean()
    if len(ew) >= 3:
        x = np.arange(len(ew[-3:])); y = ew[-3:].values
        try: slope = np.polyfit(x, y, 1)[0]
        except Exception: slope = 0.0
    else: slope = 0.0
    last = ew.iloc[-1]
    return [last + slope*(i+1) for i in range(steps)]

with tab2:
    st.markdown("#### Exposition par échéance")
    matur_colcfg = {"Metric": st.column_config.TextColumn("Metric", help="Type d’exposition (Flat price, White premium, NY/LDN spreads, LDN, NY, Futures, Option Delta).")}
    for c in matur_acc.columns:
        if c != "Metric":
            matur_colcfg[c] = st.column_config.NumberColumn(c, help="Échéance (valeur par maturité).", format="%.2f")
    st.dataframe(matur_acc, use_container_width=True, hide_index=True, column_config=matur_colcfg)

    m_f = m_long[m_long["Metric"].isin(metrics_sel)].copy()

    def parse_ech(e):
        s = str(e).replace("févr","fev").replace("janv","jan").replace("avr","apr").replace("déc","dec")
        try: return pd.to_datetime(s, dayfirst=True, errors="coerce")
        except Exception: return pd.NaT

    m_f["t"] = m_f["Echeance"].map(parse_ech)
    if m_f["t"].notna().any(): m_f.sort_values("t", inplace=True)

    explain_metric = {
        "Flat price": EXPLAIN["Flat price"], "White premium": EXPLAIN["White premium"],
        "NY spread": EXPLAIN["NY spread"],   "LDN spread": EXPLAIN["LDN spread"],
        "LDN": EXPLAIN["LDN"],               "NY": EXPLAIN["NY"],
        "Futures": EXPLAIN["Futures"],       "Option Delta": EXPLAIN["Option Delta"],
    }

    for metric in metrics_sel:
        sub = m_f[m_f["Metric"]==metric].dropna(subset=["Valeur"]).copy()
        if sub.empty:
            st.info(f"Pas de données pour **{metric}**."); continue
        fc = ewm_forecast(sub["Valeur"].values, steps=forecast_steps, alpha=alpha_ewm)

        if sub["t"].notna().all():
            last_date = sub["t"].dropna().max()
            future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(len(fc))]
            x_fc, x_hist, x_label = future_dates, sub["t"], "Date"
        else:
            x_fc = list(range(len(sub), len(sub)+len(fc))); x_hist = list(range(len(sub))); x_label="Index"

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_hist, y=sub["Valeur"], mode="lines+markers", name=f"{metric} (hist)",
            hovertemplate=f"<b>{metric}</b><br>{explain_metric.get(metric,'')}<br>%{{x}} → %{{y:.2f}}<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=x_fc, y=fc, mode="lines+markers", name=f"{metric} (forecast)",
            hovertemplate=f"<b>{metric} – Prévision</b><br>{explain_metric.get(metric,'')}<br>%{{x}} → %{{y:.2f}}<extra></extra>"
        ))
        fig2.update_layout(title=f"{metric} – Historique & Prévision",
                           xaxis_title=x_label, yaxis_title="kUSD / ratio",
                           height=380, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ==== TAB 3: ALERTING ====
with tab3:
    st.markdown("#### Alerte limites")
    g2 = g_applied.copy()
    alert_rows = g2[g2["Status"].isin(["🟠 Surv.", "🔴 Critique"])].copy()
    if alert_rows.empty:
        st.success("Aucune alerte active.")
    else:
        st.warning(f"{len(alert_rows)} alerte(s) active(s).")
        st.dataframe(
            alert_rows[["Poste","Value","Limit_applied","Usage %","Status","Warn_applied","Crit_applied"]]
              .rename(columns={"Limit_applied":"Limit (appl.)"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Poste": st.column_config.TextColumn("Poste", help="Poste de risque concerné."),
                "Value": st.column_config.NumberColumn("Value", help=EXPLAIN["Value"], format="%.2f"),
                "Limit (appl.)": st.column_config.NumberColumn("Limit (appl.)", help="Limite utilisée pour le calcul.", format="%.2f"),
                "Usage %": st.column_config.NumberColumn("Usage (%)", help=EXPLAIN["Usage"], format="%.1f"),
                "Warn_applied": st.column_config.NumberColumn("Alerte (poste)", help="Seuil d'alerte appliqué.", format="%.2f"),
                "Crit_applied": st.column_config.NumberColumn("Critique (poste)", help="Seuil critique appliqué.", format="%.2f"),
                "Status": st.column_config.TextColumn("Status", help="Surveillance ou Critique selon seuils appliqués.")
            }
        )

    st.markdown("#### Heatmap usages")
    heat = g2[["Poste","Usage"]].copy()
    heat["Usage %"] = (heat["Usage"]*100).round(1)
    fig3 = px.treemap(heat, path=['Poste'], values="Usage %", color="Usage",
                      color_continuous_scale=["#2c7","orange","red"], range_color=[0,1.2])
    fig3.update_layout(height=420, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ==== TAB 4: SENSITIVITIES ====
with tab4:
    st.markdown("#### Sensibilités & limites")
    sens_colcfg = {
        "Parameters": st.column_config.TextColumn("Parameters", help=EXPLAIN["Parameters"]),
        "Position": st.column_config.NumberColumn("Position", help=EXPLAIN["Position"], format="%.2f"),
        "Risk limits": st.column_config.NumberColumn("Risk limits", help=EXPLAIN["Risk limits"], format="%.2f"),
        "Usage": st.column_config.NumberColumn("Usage", help=EXPLAIN["Usage_sens"], format="%.2f"),
    }
    for opt_col in ["Option Delta","Gamma (lots/pt)","Vega ($/vol.pt)","CSO (lots)"]:
        if opt_col in sens_acc.columns:
            sens_colcfg[opt_col] = st.column_config.NumberColumn(opt_col, help=EXPLAIN.get(opt_col,""), format="%.2f")
    st.dataframe(sens_acc, use_container_width=True, hide_index=True, column_config=sens_colcfg)

    s_plot = sens_acc.copy()
    if "Usage" in s_plot.columns:
        s_plot["Usage %"] = (s_plot["Usage"]*100).round(1)
        fig4 = px.bar(s_plot, x="Parameters", y="Usage %", text="Usage %", title="Usage des sensibilités (%)",
                      hover_data={"Position":":.2f","Risk limits":":.2f","Usage":":.2%"})
        fig4.update_traces(textposition="outside", cliponaxis=False)
        fig4.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig4, use_container_width=True)


    # === SCÉNARIOS Δ–Γ–VEGA (P&L) ===
    st.markdown("### Scénarios Δ–Γ–Vega (P&L)")

    # --- 1) Récup Greeks depuis le tableau (si présents) ---
    def _row_val(df, row_name, col="Position"):
        try:
            return float(df.loc[df["Parameters"].astype(str).str.strip().str.lower()==row_name.lower(), col].iloc[0])
        except Exception:
            return None

    # Essais de lecture auto
    gamma_lots_pt = _row_val(sens_acc, "Gamma (lots/pt)")
    vega_usd_vol  = _row_val(sens_acc, "Vega ($/vol.pt)")
    # Δ : souvent ailleurs ; on tente sinon on demande
    delta_lots = _row_val(sens_acc, "Option Delta")
    if delta_lots is None:
        delta_lots = 0.0

    with st.expander("Paramètres scénario", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            delta_lots = st.number_input("Delta (lots équiv.)", value=float(delta_lots or 0.0), step=10.0,
                                         help="Exposition linéaire équivalente en lots (Δ total du portefeuille).")
        with c2:
            gamma_lots_pt = st.number_input("Gamma (lots/pt)", value=float(gamma_lots_pt or 0.0), step=1.0,
                                            help="Variation du Delta (en lots) par 1 point de prix.")
        with c3:
            vega_usd_vol = st.number_input("Vega ($/vol.pt)", value=float(vega_usd_vol or 0.0), step=1000.0,
                                           help="Variation de valeur (USD) pour +1 point de volatilité.")

        c4, c5, c6 = st.columns(3)
        with c4:
            point_value = st.number_input("Point value ($ / lot / pt prix)", value=1000.0, step=100.0,
                                          help="USD gagnés/perdus par lot pour 1 point de prix. À ajuster selon le contrat.")
        with c5:
            price_range = st.slider("ΔPrix (points)", -10.0, 10.0, (-5.0, 5.0), 0.5,
                                    help="Plage des variations de prix testées (points).")
        with c6:
            vol_range = st.slider("ΔVol (points)", -10.0, 10.0, (-5.0, 5.0), 0.5,
                                  help="Plage des variations de volatilité implicite (points).")

    # --- 2) Grille de scénarios & P&L ---
    dS = np.arange(price_range[0], price_range[1] + 1e-9, 0.5)  # prix (pts)
    dV = np.arange(vol_range[0],  vol_range[1]  + 1e-9, 0.5)   # vol (pts)

    # P&L = Δ * pv * dS  + 0.5 * Γ * pv * dS^2  +  Vega * dV
    # où pv = point_value ($ / lot / pt)
    pv = float(point_value)
    dS_grid, dV_grid = np.meshgrid(dS, dV, indexing="xy")
    pnl_grid = (delta_lots * pv * dS_grid) + (0.5 * gamma_lots_pt * pv * (dS_grid ** 2)) + (vega_usd_vol * dV_grid)

    st.markdown("---")
    st.markdown("#### Formule du stress test (Δ–Γ–Vega)")

    st.latex(r"P\&L = \Delta \cdot pv \cdot \Delta S \;+\; \tfrac{1}{2}\Gamma \cdot pv \cdot (\Delta S)^2 \;+\; Vega \cdot \Delta V")

    st.markdown("""
    **Décomposition rapide (chaque bloc de la formule) :**
    - **Δ (Delta)** × **pv (point value)** × **ΔS (variation du prix)**  
      → Cela mesure l’impact **linéaire et direct** d’un mouvement du prix.  
      Exemple : si Δ = 200 lots, pv = 1 000 $/pt, ΔS = +2 pts → **+400 000 $**.
    
    - **0.5 × Γ (Gamma)** × **pv** × **(ΔS)²**  
      → Cela capture l’impact de la **courbure** : ton Delta change quand le prix bouge beaucoup.  
      Exemple : Γ = 50 lots/pt, pv = 1 000 $/pt, ΔS = +2 pts → **+100 000 $**.
    
    - **Vega × ΔV (variation de volatilité)**  
      → Cela mesure la sensibilité à la **volatilité implicite** du marché.  
      Exemple : Vega = 40 000 $/pt, ΔV = –3 pts → **–120 000 $**.
    """)

    st.caption("👉 On additionne ces trois blocs pour obtenir le P&L approximatif du portefeuille dans chaque scénario de stress.")
    st.markdown("---")



    # --- 3) Heatmap & résumé worst/best ---
    hm = px.imshow(
        pnl_grid,
        x=dS, y=dV,
        origin="lower",
        aspect="auto",
        labels=dict(x="ΔPrix (pts)", y="ΔVol (pts)", color="P&L (USD)"),
        title="Heatmap scénario Δ–Γ–Vega — P&L (USD)"
    )
    st.plotly_chart(hm, use_container_width=True)

    # Résumés
    worst_idx = np.unravel_index(np.argmin(pnl_grid), pnl_grid.shape)
    best_idx  = np.unravel_index(np.argmax(pnl_grid), pnl_grid.shape)
    worst_pnl = float(pnl_grid[worst_idx])
    best_pnl  = float(pnl_grid[best_idx])
    worst_ds, worst_dv = float(dS[worst_idx[1]]), float(dV[worst_idx[0]])
    best_ds,  best_dv  = float(dS[best_idx[1]]),  float(dV[best_idx[0]])

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Pire scénario (USD)", f"{worst_pnl:,.0f}",
                  help=f"à ΔPrix={worst_ds:+.1f} pt ; ΔVol={worst_dv:+.1f} pt")
    with cB:
        st.metric("Meilleur scénario (USD)", f"{best_pnl:,.0f}",
                  help=f"à ΔPrix={best_ds:+.1f} pt ; ΔVol={best_dv:+.1f} pt")
    with cC:
        st.metric("Asymétrie (Best - Worst)", f"{(best_pnl - worst_pnl):,.0f}")

    # --- MINI-VIZ : Linéaire (Δ) vs Quadratique (Δ + Γ) ---
    with st.expander("Comparer linéaire (Δ) vs quadratique (Δ + Γ)", expanded=True):
        # ΔVol = 0 pour isoler l'effet prix (Vega exclu)
        ds_cmp = np.arange(price_range[0], price_range[1] + 1e-9, 0.5)
        pv = float(point_value)

        # Calcul des deux versions
        pnl_linear = (delta_lots * pv * ds_cmp)
        pnl_quadratic = (delta_lots * pv * ds_cmp) + (0.5 * gamma_lots_pt * pv * (ds_cmp ** 2))

        # Graphique comparatif
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=ds_cmp, y=pnl_linear, mode="lines+markers",
            name="Δ seul (linéaire)",
            hovertemplate="ΔPrix=%{x:+.1f} pt<br>P&L=%{y:,.0f} $<extra></extra>"
        ))
        fig_cmp.add_trace(go.Scatter(
            x=ds_cmp, y=pnl_quadratic, mode="lines+markers",
            name="Δ + Γ (quadratique)",
            hovertemplate="ΔPrix=%{x:+.1f} pt<br>P&L=%{y:,.0f} $<extra></extra>"
        ))

        fig_cmp.update_layout(
            title="Δ vs Δ+Γ — Impact du prix (ΔVol = 0)",
            xaxis_title="ΔPrix (points)",
            yaxis_title="P&L (USD)",
            height=340,
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.caption("Δ = pente (linéaire). Γ ajoute la courbure (terme 0.5·Γ·pv·(ΔS)²). ΔVol=0 ⇒ Vega exclu pour comparaison propre.")

    # Courbe coupe ΔVol=0 (impact du prix seul)
    try:
        zero_vol_row = np.where(np.isclose(dV, 0.0))[0][0]
        line_price = go.Figure()
        line_price.add_trace(go.Scatter(
            x=dS, y=pnl_grid[zero_vol_row, :], mode="lines+markers",
            name="ΔVol = 0",
            hovertemplate="ΔPrix=%{x:+.1f} → P&L=%{y:,.0f}$<extra></extra>"
        ))
        line_price.update_layout(
            title="Coupe ΔVol = 0 — Effet prix seul (Δ + Γ)",
            xaxis_title="ΔPrix (pts)",
            yaxis_title="P&L (USD)",
            height=320
        )
        st.plotly_chart(line_price, use_container_width=True)
    except Exception:
        pass


    

    # Courbe coupe ΔVol=0 (impact du prix seul)
    try:
        zero_vol_row = np.where(np.isclose(dV, 0.0))[0][0]
        line_price = go.Figure()
        line_price.add_trace(go.Scatter(x=dS, y=pnl_grid[zero_vol_row, :], mode="lines+markers",
                                        name="ΔVol = 0", hovertemplate="ΔPrix=%{x:+.1f} → P&L=%{y:,.0f}$<extra></extra>"))
        line_price.update_layout(title="Coupe ΔVol = 0 — Effet prix seul (Δ + Γ)", xaxis_title="ΔPrix (pts)", yaxis_title="P&L (USD)", height=320)
        st.plotly_chart(line_price, use_container_width=True)
    except Exception:
        pass
        

# ==== TAB 5: METHODO ====
with tab5:
    st.markdown("### 📘 Méthodologie — en bref")
    st.markdown("""
- **Limites par poste** : sélectionnez un poste → définissez **Limit**, **Alerte**, **Critique**, et **Apply**.
- **Calcul** : `Usage = Value / Limit_applied` si *Apply* est ON. Sinon, Usage = vide.
- **Seuils** : on applique d’abord les seuils **du poste** ; à défaut, `Alerte=0.80` et `Critique=1.00`.
- **Prévision (EWMA)** : lissage exponentiel (α paramétrable). Projection courte via pente des 3 derniers points lissés.
- **Fusion** : lecture `template1.xlsx`…`templateN.xlsx` avec règle **last wins** et union colonnes/lignes.
""")

# --------------------- TOASTS ALERTS ---------------------
if g_applied["Usage"].notna().any() and (g_applied["Usage"] >= g_applied["Crit_applied"]).any():
    st.toast("Dépassement limite critique détecté.", icon="🚨")
elif g_applied["Usage"].notna().any() and (g_applied["Usage"] >= g_applied["Warn_applied"]).any():
    st.toast("Surveillance: certains postes approchent la limite.", icon="⚠️")
else:
    st.toast("Risque global maîtrisé.", icon="✅")
