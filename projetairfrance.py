# Streamlit – Air France-KLM: Liquidity, Solvency & Credit Rating Sandbox
# ----------------------------------------------------------------------
# Ajouts:
# - Note de crédit simulée (A / BBB / B / CCC)
# - Graphiques interactifs (Altair .interactive())
# - Année de simulation "SIM" : l'utilisateur ajuste 3 leviers clés
#   (Current ratio, Net debt/EBITDA, Interest coverage) et la note s'adapte.
# ----------------------------------------------------------------------

import json
from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="AF-KLM – Liquidity, Solvency & Credit Rating",
    page_icon="",
    layout="wide",
)

# ================== DATA BASE ==================
BASE_DATA = {
    "Liquidity": {
        "Current ratio": {"2023": 0.74, "2024": 0.65},
        "Quick ratio": {"2023": 0.69, "2024": 0.59},
        "Acid test": {"2023": 0.65, "2024": 0.55},
        "Cash ratio": {"2023": 0.39, "2024": 0.30},
    },
    "Solvency": {
        "Equity ratio (Equity/Assets)": {"2023": 0.01, "2024": 0.02},
        "Debt-to-assets": {"2023": 0.39, "2024": 0.39},
        "Net debt / EBITDA (x)": {"2023": 1.20, "2024": 1.73},
    },
    "Coverage": {
        "Interest coverage (EBIT / interest)": {"2023": 4.83, "2024": 4.47},
        "Cash interest cov. (EBITDA / interest)": {"2023": 5.98, "2024": 6.39},
    },
}

# ================== HELPERS ==================

def dict_to_df(d: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for section, metrics in d.items():
        for metric, vals in metrics.items():
            row = {"Section": section, "Metric": metric}
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows)


def df_to_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for _, row in df.iterrows():
        section = row["Section"]
        metric = row["Metric"]
        vals = {c: row[c] for c in df.columns if c not in ["Section", "Metric"]}
        out.setdefault(section, {})[metric] = vals
    return out


def compute_credit_rating(df: pd.DataFrame, year: str) -> str:
    """Attribue une note simple selon 3 critères clés (liquidité, levier, couverture)."""
    # Extraction sécurisée
    def _get(metric_name: str):
        try:
            return float(df.loc[df["Metric"] == metric_name, year].values[0])
        except Exception:
            return np.nan

    current_ratio = _get("Current ratio")
    net_debt_ebitda = _get("Net debt / EBITDA (x)")
    interest_cov = _get("Interest coverage (EBIT / interest)")

    if np.isnan(current_ratio) or np.isnan(net_debt_ebitda) or np.isnan(interest_cov):
        return "N/A"

    score = 0
    # Règles simplifiées
    if current_ratio >= 1.0:  # liquidité minimale confortable
        score += 1
    if net_debt_ebitda <= 2.0:  # levier modéré
        score += 1
    if interest_cov >= 5.0:  # couverture des intérêts robuste
        score += 1

    if score == 3:
        return "A (Solide)"
    elif score == 2:
        return "BBB (Correct)"
    elif score == 1:
        return "B (Tendu)"
    else:
        return "CCC (Fragile)"


# ================== SESSION INIT ==================
if "base" not in st.session_state:
    st.session_state.base = deepcopy(BASE_DATA)
if "scenario" not in st.session_state:
    st.session_state.scenario = deepcopy(BASE_DATA)

st.title(" Air France–KLM – Liquidity & Solvency + Credit Rating")

scenario_df = dict_to_df(st.session_state.scenario)

# ================== ÉDITEUR ==================
edited = st.data_editor(
    scenario_df,
    use_container_width=True,
    num_rows="dynamic",
    key="editor",
)

# ---------- Bloc SIMULATION ----------
st.sidebar.header(" Année de simulation (SIM)")
use_sim = st.sidebar.checkbox("Activer l'année SIM", value=True)

# Valeurs par défaut pour SIM = copie de la dernière année dispo (2024)
last_year = max([c for c in edited.columns if c not in ["Section", "Metric"]], default="2024")

default_current = float(edited.loc[edited["Metric"] == "Current ratio", last_year].values[0]) if "Current ratio" in edited["Metric"].values else 0.8
default_nd_ebitda = float(edited.loc[edited["Metric"] == "Net debt / EBITDA (x)", last_year].values[0]) if "Net debt / EBITDA (x)" in edited["Metric"].values else 2.0
default_ic = float(edited.loc[edited["Metric"] == "Interest coverage (EBIT / interest)", last_year].values[0]) if "Interest coverage (EBIT / interest)" in edited["Metric"].values else 4.0

if use_sim:
    st.sidebar.caption("Ajuste les leviers pour tester l'impact sur la note de crédit.")
    sim_current = st.sidebar.number_input("Current ratio (SIM)", value=float(default_current), step=0.05, min_value=0.0)
    sim_nd_ebitda = st.sidebar.number_input("Net debt / EBITDA (x) (SIM)", value=float(default_nd_ebitda), step=0.1, min_value=0.0)
    sim_ic = st.sidebar.number_input("Interest coverage (EBIT / interest) (SIM)", value=float(default_ic), step=0.1, min_value=0.0)

    # Appliquer une colonne "SIM" dans le DF édité
    if "SIM" not in edited.columns:
        edited["SIM"] = np.nan

    # Copier par défaut la dernière année, puis écraser les 3 indicateurs clés
    for idx, row in edited.iterrows():
        metric = row["Metric"]
        if pd.isna(edited.at[idx, "SIM"]) and last_year in edited.columns:
            edited.at[idx, "SIM"] = edited.at[idx, last_year]
        # Écrasement ciblé
        if metric == "Current ratio":
            edited.at[idx, "SIM"] = sim_current
        elif metric == "Net debt / EBITDA (x)":
            edited.at[idx, "SIM"] = sim_nd_ebitda
        elif metric == "Interest coverage (EBIT / interest)":
            edited.at[idx, "SIM"] = sim_ic

# Reconstruire le dict scénario (intègre SIM si activé)
st.session_state.scenario = df_to_dict(edited)

# ================== CHOIX ANNÉE ==================
existing_years = [c for c in edited.columns if c not in ["Section", "Metric"]]

if existing_years:
    year_default_index = existing_years.index("SIM") if "SIM" in existing_years else len(existing_years) - 1
    year_pick = st.selectbox("Année à analyser", options=existing_years, index=year_default_index)

    st.subheader(f"Ratios financiers – {year_pick}")
    st.dataframe(edited[["Section", "Metric", year_pick]], use_container_width=True)

    # Note de crédit
    note = compute_credit_rating(edited, year_pick)
    st.markdown(f"### 🏦 Note de crédit simulée ({year_pick}) : **{note}**")

# ================== GRAPHIQUES INTERACTIFS ==================
st.subheader("Graphiques interactifs")

all_metrics = edited["Metric"].unique().tolist()
metrics_sel = st.multiselect(
    "Choisir un ou plusieurs indicateurs",
    options=all_metrics,
    default=min(3, len(all_metrics)) and all_metrics[:3],
)

if metrics_sel:
    viz = edited.melt(id_vars=["Section", "Metric"], var_name="Year", value_name="Value")
    viz = viz[viz["Metric"].isin(metrics_sel)]
    chart = (
        alt.Chart(viz)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:N", title="Année"),
            y=alt.Y("Value:Q", title="Valeur"),
            color=alt.Color("Metric:N", title="Indicateur"),
            tooltip=["Section", "Metric", "Year", alt.Tooltip("Value:Q", format=".2f")],
        )
        .interactive()  # zoom/pan + légende cliquable
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Sélectionne au moins un indicateur pour tracer le graphique.")
