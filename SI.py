# -*- coding: utf-8 -*-
# Odoo — Management & Audit des SI (PhD) — v2.4
# 4 jours • sous-onglets détaillés • PyVis (graphes interactifs, tooltips) • Graphviz (BPMN)
# Plotly (KPI, SLA, Heatmap risques) • Export PDF long (texte + visuels) • Thème dark accessible
# Calculateur ROI/NPV/Payback • Calculateur SLA/SLO & Error Budget
#
# Lancer localement :  streamlit run app_odoo_si_phd.py
#
# NOTE IMPORTANTE (données) :
# Les chiffres par défaut sont didactiques (simulés) et doivent être branchés à vos tables Odoo
# (sale_order, account_move, stock_move, res_partner, stock_picking, etc.) pour un usage réel.

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import textwrap, tempfile, os
from typing import List, Dict

# Graphs
from pyvis.network import Network
import streamlit.components.v1 as components
from graphviz import Digraph

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Odoo — Management & Audit des SI", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------------------
# THEME / CSS — Dark accessible (contraste AA)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
  --bg:#0a0f1d;
  --panel:#0f172a;
  --muted:#a5b4c2;
  --border:#1e293b;
  --accent:#38bdf8;
  --accent2:#a78bfa;
  --ok:#22c55e;
  --warn:#f59e0b;
  --danger:#ef4444;
  --fg:#e5e7eb;
}
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }
h1,h2,h3 { margin: 0.2rem 0 0.6rem 0; color: var(--fg); }
hr { height:1px; border:0; background: linear-gradient(90deg,var(--accent),transparent); margin: 8px 0 18px; }
.block { background: var(--panel); color: var(--fg); border-radius: 16px; padding: 16px 18px; border: 1px solid var(--border); }
.small { color: var(--muted); font-size: 0.92rem; }
.kpi { background:#0c1426; border:1px solid var(--border); border-radius:16px; padding:14px; text-align:center; }
.kpi .v { font-size:1.7rem; font-weight:800; color:#fff; }
.kpi .l { color: var(--muted); }
.stTabs [data-baseweb="tab-list"] { gap: 10px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: #0c1426; border-radius: 12px; padding: 10px 16px; border: 1px solid var(--border); color:#dbe2ea; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { border-color: var(--accent); box-shadow: inset 0 -3px 0 var(--accent); color: #fff; }
.stButton>button[kind="primary"] { background: var(--accent)!important; border-color: var(--accent)!important; color:#06121f!important; font-weight:700; }
.badge { display:inline-flex; align-items:center; gap:.4rem; padding:.15rem .55rem; border-radius:10px; border:1px solid var(--border); background:#0c1426; color:#cfd6df; margin:.15rem .3rem .15rem 0; }
.badge.OK{border-color:#1d3b2a;color:#a7f3d0;background:#052e1a;}
.badge.WARN{border-color:#3b2d05;color:#fde68a;background:#1f1501;}
.badge.KO{border-color:#4b1111;color:#fecaca;background:#1f0a0a;}
table { font-size: .95rem; }
a { color: #5dd3ff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DONNÉES DEMO COHÉRENTES (remplaçables par vos données Odoo)
# ---------------------------------------------------------------------------
np.random.seed(42)
YEARS = np.array([2021,2022,2023,2024,2025,2026])
revenus = np.array([42,55,61,74,88,96])   # M€
ebitda  = np.array([ 9,12,13,15,18,20])   # M€
capex   = np.array([ 3, 4, 4, 5, 6, 6])   # M€
opex    = np.array([12,14,15,16,18,19])   # M€
incidents = np.array([22,18,15,12, 9, 8])
lat_p95 = np.array([520,470,430,390,360,340])   # ms
err_rate = np.array([0.80,0.70,0.60,0.45,0.35,0.30])  # %
marge = np.round(100*ebitda/revenus,1)

df_fin = pd.DataFrame({
    "Année": YEARS,
    "Revenus (M€)": revenus,
    "EBITDA (M€)": ebitda,
    "CAPEX (M€)": capex,
    "OPEX (M€)": opex,
    "Incidents IT (nb)": incidents,
    "P95 Latence (ms)": lat_p95,
    "Error rate (%)": err_rate,
    "Marge EBITDA (%)": marge
})

modules = ["ERP Core","Comptabilité","CRM","Achats","Stock","RH","Projet","E-commerce"]
sla = np.clip(np.random.normal(99.74,0.18,len(modules)), 99.0, 99.95).round(2)
df_sla = pd.DataFrame({"Module":modules,"SLA (%)":sla}).sort_values("SLA (%)", ascending=False)

# RCM priorisée (exemple)
rcm = pd.DataFrame({
    "ID":["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10"],
    "Domaine":["Accès","Données","Dispo","Conformité","Cyber","Intégrations","Changements","Fournisseurs","Finance","Fraude"],
    "Risque":[
        "Droits excessifs / comptes orphelins",
        "Données inexactes / référentiels incohérents",
        "Panne infra / RTO>2h / RPO>30min",
        "CNIL/RGPD (base légale, purge, droits)",
        "Ransomware / clés exposées / vuln applicatives",
        "Webhooks/API mal filtrés (DoS / injection)",
        "Déploiement non maîtrisé / rollback impossible",
        "SaaS critique indispo / réversibilité",
        "Écritures comptables erronées / clôtures",
        "Remises commerciales abusives / contournement"
    ],
    "Probabilité":[3,3,2,2,3,3,3,2,2,2],
    "Impact":[4,4,5,4,5,4,4,4,4,4],
    "Contrôle clé":[
        "RBAC, recertif Q, SoD, JML automatisé",
        "MDM, contrôles ETL, matching référentiels, DQ rules",
        "DRP testé, réplication multi-zone, monitoring SLO",
        "Registre, DPIA, portabilité/suppression, purge",
        "EDR, KMS, patching, WAF, tests intrusion",
        "Auth, rate-limit, schema validation, allow-list",
        "CAB/ITIL, blue-green, runbook rollback, tests",
        "SLA contractuel, plan continuité, escrow, sorties",
        "Four-eyes, clôture/verrou, séquences non ruptibles",
        "Double validation, seuils, piste d’audit"
    ],
    "Owner":["DSI","DAF/DSI","Infra","DPO","SOC/DSI","DSI/Intégrateur","Change Manager","Achat/DSI","DAF/Compta","Ventes/DAF"]
})
rcm["Score"] = rcm["Probabilité"] * rcm["Impact"]

bsc = pd.DataFrame({
    "Perspective":["Financière","Client","Processus","Apprentissage"],
    "KPI":["ROI ERP (%)","Satisfaction (/10)","SLA≥99.5% (mois)","Formations/an"],
    "2023":[11.0,7.2,10,6],
    "2024":[14.5,7.8,11,8],
    "2025":[16.0,8.2,12,10]
})

# ---------------------------------------------------------------------------
# OUTILS CALCUL — ROI/NPV/PAYBACK & SLA
# ---------------------------------------------------------------------------
def npv(rate: float, cashflows: List[float]) -> float:
    return float(sum(cf/((1+rate)**i) for i, cf in enumerate(cashflows)))

def irr(cashflows: List[float], guess: float=0.1, tol: float=1e-6, max_iter: int=100):
    r = guess
    for _ in range(max_iter):
        f = sum(cf/((1+r)**i) for i, cf in enumerate(cashflows))
        df = sum(-i*cf/((1+r)**(i+1)) for i, cf in enumerate(cashflows))
        if abs(df) < 1e-12:
            break
        r_new = r - f/df
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
    return None

def payback_period(cashflows: List[float]):
    cum = 0.0
    for i, cf in enumerate(cashflows):
        cum += cf
        if cum >= 0:
            prev = cum - cf
            if cf == 0:
                return float(i)
            frac = (0 - prev) / cf
            return i - 1 + frac
    return None

def error_budget_minutes(sla_target: float, days: int=30) -> int:
    minutes = days*24*60
    return int(round((1 - sla_target/100.0) * minutes))

# ---------------------------------------------------------------------------
# PLOTLY HELPERS
# ---------------------------------------------------------------------------
def fig_revenus_ebitda():
    f=go.Figure()
    f.add_trace(go.Scatter(x=YEARS,y=revenus,mode="lines+markers",name="Revenus (M€)",
                           hovertemplate="Année %{x}<br>Revenus %{y} M€<extra></extra>"))
    f.add_trace(go.Scatter(x=YEARS,y=ebitda,mode="lines+markers",name="EBITDA (M€)",
                           hovertemplate="Année %{x}<br>EBITDA %{y} M€<extra></extra>"))
    f.update_layout(height=420, template="plotly_white", title="Trajectoire Revenus & EBITDA",
                    xaxis_title="Année", yaxis_title="M€")
    return f

def fig_capex_opex():
    f=go.Figure()
    f.add_trace(go.Bar(x=YEARS, y=capex, name="CAPEX"))
    f.add_trace(go.Bar(x=YEARS, y=opex,  name="OPEX"))
    f.update_layout(barmode="stack", height=360, template="plotly_white",
                    title="CAPEX vs OPEX", xaxis_title="Année", yaxis_title="M€")
    return f

def fig_incidents():
    f=go.Figure(go.Bar(x=YEARS, y=incidents, name="Incidents IT",
                       hovertemplate="Année %{x}<br>Incidents %{y}<extra></extra>"))
    f.update_layout(height=360, template="plotly_white", title="Incidents IT (par an)",
                    xaxis_title="Année", yaxis_title="Nb")
    return f

def fig_latency_error():
    f=go.Figure()
    f.add_trace(go.Scatter(x=YEARS, y=lat_p95, mode="lines+markers", name="P95 Latence (ms)"))
    f.add_trace(go.Scatter(x=YEARS, y=err_rate, mode="lines+markers", name="Taux d'erreur (%)", yaxis="y2"))
    f.update_layout(
        height=360, template="plotly_white", title="Perf applicative (P95 & erreurs)",
        xaxis=dict(title="Année"),
        yaxis=dict(title="P95 (ms)"),
        yaxis2=dict(title="Erreurs (%)", overlaying="y", side="right")
    )
    return f

def fig_sla_bar():
    f=go.Figure(go.Bar(x=df_sla["Module"], y=df_sla["SLA (%)"],
                       text=df_sla["SLA (%)"], textposition="auto",
                       hovertemplate="Module %{x}<br>SLA %{y}%<extra></extra>"))
    f.update_layout(height=430, template="plotly_white", title="SLA par module",
                    xaxis_title="Module", yaxis_title="SLA (%)", yaxis_range=[98.8,100])
    return f

def fig_roi_series(vals_years: List[int], vals: List[float], title="ROI ERP (%)"):
    f=go.Figure(go.Scatter(x=vals_years, y=vals, mode="lines+markers",
                           hovertemplate="Année %{x}<br>ROI %{y}%<extra></extra>"))
    f.update_layout(height=320, template="plotly_white", title=title)
    return f

def fig_capacity_diurnal():
    hours=list(range(24))
    req=np.round(np.clip(np.random.normal(800,240,24), 250, 1600)).astype(int)
    f=go.Figure(go.Bar(x=hours,y=req))
    f.update_layout(height=320, template="plotly_white", title="Charge applicative (req/min)",
                    xaxis_title="Heure", yaxis_title="Requêtes/min")
    return f

def fig_risk_heatmap(rcm_df: pd.DataFrame):
    pivot = pd.DataFrame(0, index=[1,2,3,4,5], columns=[1,2,3,4,5])
    for _, r in rcm_df.iterrows():
        pivot.loc[r["Impact"], r["Probabilité"]] += 1
    pivot = pivot.sort_index(ascending=True)
    f = px.imshow(pivot, text_auto=True, color_continuous_scale="Reds", aspect="auto")
    f.update_layout(height=420, template="plotly_white", title="Heatmap Risques (nb par case Impact×Probabilité)",
                    xaxis_title="Probabilité", yaxis_title="Impact")
    return f

# ---------------------------------------------------------------------------
# Graphviz — BPMN simplifié (ordre des étapes et gateways)
# ---------------------------------------------------------------------------
def bpmn_o2c_graph() -> Digraph:
    g = Digraph("O2C", comment="Order-to-Cash", format="svg")
    g.attr(rankdir="LR", bgcolor="#0a0f1d", fontcolor="white")
    g.attr("node", shape="rectangle", style="rounded,filled", color="#1e293b", fillcolor="#0f172a", fontname="Inter", fontcolor="white")
    g.attr("edge", color="#94a3b8")
    steps = [
        ("c","Client (Commande)"),
        ("q","Devis / Offre (Sales)"),
        ("s","Commande client (SO)"),
        ("p","Préparation / Picking (Stock)"),
        ("d","Expédition (Delivery)"),
        ("i","Facture client (AR)"),
        ("r","Règlement / Lettrage (Treasury)")
    ]
    for k, lbl in steps: g.node(k, lbl)
    edges = [("c","q"),("q","s"),("s","p"),("p","d"),("d","i"),("i","r")]
    for a,b in edges: g.edge(a,b, arrowhead="normal")
    return g

def bpmn_p2p_graph() -> Digraph:
    g = Digraph("P2P", comment="Procure-to-Pay", format="svg")
    g.attr(rankdir="LR", bgcolor="#0a0f1d", fontcolor="white")
    g.attr("node", shape="rectangle", style="rounded,filled", color="#1e293b", fillcolor="#0f172a", fontname="Inter", fontcolor="white")
    g.attr("edge", color="#94a3b8")
    steps = [
        ("r","Demande d'achat"),
        ("po","Bon de commande (PO)"),
        ("gr","Réception (GRN)"),
        ("vb","Facture fournisseur"),
        ("ap","Paiement (AP)"),
        ("recon","Rapprochement bancaire")
    ]
    for k,lbl in steps: g.node(k,lbl)
    edges = [("r","po"),("po","gr"),("gr","vb"),("vb","ap"),("ap","recon")]
    for a,b in edges: g.edge(a,b, arrowhead="normal")
    return g

# ---------------------------------------------------------------------------
# PyVis — schémas interactifs (architecture, gouvernance, réseau)
# ---------------------------------------------------------------------------
def pyvis_graph(nodes: List[Dict], edges: List[Dict], height: str="560px", physics: bool=True):
    net = Network(height=height, width="100%", bgcolor="#0b0f1a", font_color="#e5e7eb", directed=True, notebook=False)
    if physics:
        net.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=180, spring_strength=0.02)
    else:
        net.toggle_physics(False)
    for n in nodes:
        net.add_node(n["id"], label=n["label"], title=n.get("title",""),
                     shape=n.get("shape","ellipse"), size=n.get("size",28),
                     color=n.get("color","#60a5fa"), borderWidth=2)
    for e in edges:
        net.add_edge(e["src"], e["dst"], title=e.get("title",""), arrows="to",
                     color=e.get("color","#94a3b8"), width=e.get("width",1))
    fd, path = tempfile.mkstemp(suffix=".html"); os.close(fd)
    net.write_html(path, open_browser=False)
    with open(path,"r",encoding="utf-8") as f:
        html=f.read()
    components.html(html, height=int(height.replace("px",""))+20, scrolling=True)

# ---------------------------------------------------------------------------
# PDF export (sections + visuels)
# ---------------------------------------------------------------------------
def mpl_from_plotly(fig, title=""):
    import matplotlib.pyplot as plt
    x=None; y=None
    for tr in fig.data:
        if hasattr(tr, "x") and hasattr(tr, "y"):
            x, y = tr.x, tr.y
            break
    g,ax=plt.subplots(figsize=(6,3.4)) 
    if x is not None and y is not None:
        ax.plot(x,y,marker="o")
    ax.grid(True); ax.set_title(title or fig.layout.title.text or "")
    buf=BytesIO(); g.savefig(buf,format="png",dpi=180,bbox_inches="tight"); plt.close(g); buf.seek(0)
    return buf.getvalue()

def mpl_bar_snapshot(fig, title=""):
    import matplotlib.pyplot as plt
    trac=None
    for tr in fig.data:
        if tr.type in ("bar","histogram"):
            trac=tr; break
    x = getattr(trac,"x",[]); y=getattr(trac,"y",[])
    g,ax=plt.subplots(figsize=(6,3.4)); ax.bar(x,y); ax.grid(True,axis="y")
    ax.set_title(title or fig.layout.title.text or "")
    buf=BytesIO(); g.savefig(buf,format="png",dpi=180,bbox_inches="tight"); plt.close(g); buf.seek(0)
    return buf.getvalue()

def export_pdf(title: str, sections: List[Dict]) -> BytesIO:
    buf=BytesIO(); c=canvas.Canvas(buf,pagesize=A4); W,H=A4; m=1.5*cm; y=H-m
    c.setFont("Helvetica-Bold",16); c.drawString(m,y,title)
    c.setFont("Helvetica",9); y-=0.6*cm; c.drawString(m,y,f"Date: {datetime.now():%Y-%m-%d %H:%M}"); y-=0.8*cm
    for s in sections:
        if y<3*cm: c.showPage(); y=H-m; c.setFont("Helvetica",9)
        c.setFont("Helvetica-Bold",12); c.drawString(m,y,s.get("title","")); y-=0.5*cm
        c.setFont("Helvetica",9)
        for line in textwrap.fill(s.get("text",""),width=105).split("\n"):
            if y<3*cm: c.showPage(); y=H-m; c.setFont("Helvetica",9)
            c.drawString(m,y,line); y-=0.42*cm
        y-=0.2*cm
        for (img,wcm) in s.get("images",[]):
            if y<6*cm: c.showPage(); y=H-m; c.setFont("Helvetica",9)
            ir=ImageReader(BytesIO(img)); iw,ih=ir.getSize(); w=wcm*cm; h=(ih/iw)*w
            c.drawImage(ir,m,y-h,width=w,height=h,preserveAspectRatio=True,mask='auto'); y-=(h+0.35*cm)
        y-=0.25*cm
    c.showPage(); c.save(); buf.seek(0); return buf

# ---------------------------------------------------------------------------
# SOURCES (multi-langues) — affichées dans l'app (liens)
# ---------------------------------------------------------------------------
SOURCES = {
    "Jour 1 — Business Case & Gouvernance": [
        ("Odoo — Case Study (Portugal, ROI)", "https://www.odoo.com"),
        ("BizzAppDev — ROI & TCO Odoo", "https://www.bizzappdev.com"),
        ("Smile (intégrateur) — maîtriser le spécifique", "https://www.smile.eu"),
        ("Gouvernance projet — COPIL / MOA / MOE (Sqorus, Everwin)", "https://www.sqorus.com, https://www.everwin.fr")
    ],
    "Jour 2 — Processus & Architecture": [
        ("O2C/P2P sous Odoo — explications (MuchConsulting, Medium)", "https://www.muchconsulting.com, https://medium.com"),
        ("Docs Odoo — ORM, modules, API JSON-RPC", "https://odoo-docs.readthedocs.io"),
        ("Architecture 3-tiers & déploiement", "https://odoo-docs.readthedocs.io")
    ],
    "Jour 3 — Données, Cyber, SLA": [
        ("Odoo Security — hashing PBKDF2, TLS, backups (Odoo)", "https://www.odoo.com"),
        ("2FA/SSO, RBAC, Portal/Public (MuchConsulting)", "https://www.muchconsulting.com"),
        ("ETL/Data Mart (Peliqan, SprintIT)", "https://www.peliqan.io, https://www.sprintit.global")
    ],
    "Jour 4 — Audit, Contrôles, Reporting": [
        ("Audit trail (Odoo 18), conformité anti-fraude (LNE)", "https://nalios.com, https://blog.tout-pour-la-gestion.com"),
        ("COBIT, ITIL — bonnes pratiques d'audit SI", "https://www.isaca.org, https://www.axelos.com"),
        ("BSC — balancedscorecard.org", "https://balancedscorecard.org")
    ]
}

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("<h1>🧭 Odoo — Management & Audit des SI (PhD)</h1>", unsafe_allow_html=True)
st.caption("Parcours interactif en 4 jours → sous-onglets. KPI dynamiques, BPMN, architectures PyVis/Graphviz, calculs ROI/SLA, RCM & Audit, export PDF.")

with st.expander("⚙️ Réglages d’affichage / options"):
    node_size = st.slider("Taille des nœuds (schémas interactifs)", 18, 62, 34, 2)
    physics   = st.toggle("Physique PyVis (forces)", value=True, help="Désactiver pour figer les positions.")
    show_sources = st.toggle("Afficher les sources sous chaque jour", value=True)

# ---------------------------------------------------------------------------
# TABS (4 DAYS)
# ---------------------------------------------------------------------------
d1, d2, d3, d4 = st.tabs([
    "Day 1 — Business Case & Governance",
    "Day 2 — Process & Architecture",
    "Day 3 — Data, Cyber & SLA",
    "Day 4 — Risks, Audit & Reporting"
])

# ===========================================================================
# DAY 1
# ===========================================================================
with d1:
    s1, s2, s3, s4, s5 = st.tabs([
        "Intro & KPI",
        "Business Case & ROI",
        "Gouvernance (acteurs/flux)",
        "Décision Rights & RACI",
        "Sources"
    ])

    with s1:
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='kpi'><div class='v'>{revenus[-1]} M€</div><div class='l'>Revenus 2026</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi'><div class='v'>{marge[-1]} %</div><div class='l'>Marge EBITDA 2026</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi'><div class='v'>{incidents[-1]}</div><div class='l'>Incidents IT 2026</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi'><div class='v'>{lat_p95[-1]} ms</div><div class='l'>P95 Latence 2026</div></div>", unsafe_allow_html=True)
        st.plotly_chart(fig_revenus_ebitda(), use_container_width=True, key="d1_re")
        st.plotly_chart(fig_capex_opex(),   use_container_width=True, key="d1_capex")
        st.plotly_chart(fig_incidents(),    use_container_width=True, key="d1_inc")
        st.plotly_chart(fig_latency_error(),use_container_width=True, key="d1_lat")

    with s2:
        st.markdown("#### Calculateur ROI / NPV / Payback")
        c1,c2,c3 = st.columns(3)
        with c1:
            capex0 = st.number_input("CAPEX initial (k€)", min_value=0, value=350, step=10, help="Investissement initial (matériel, intégration, formation).")
            opex_y = st.number_input("OPEX Annuel (k€)", min_value=0, value=95, step=5, help="Abonnements, hébergement, support.")
        with c2:
            savings_y = st.number_input("Gains annuels (k€)", min_value=0, value=210, step=5, help="Économies + marge additionnelle.")
            horizon = st.slider("Horizon (années)", 1, 8, 5)
        with c3:
            rate = st.number_input("Taux actualisation (WACC, %)", min_value=0.0, value=8.0, step=0.5) / 100.0
            res_placeholder = st.empty()

        cash = [-capex0] + [(savings_y - opex_y)]*horizon
        npv_val = npv(rate, cash)
        irr_val = irr(cash) or float("nan")
        pb = payback_period(cash)

        with res_placeholder.container():
            b1,b2,b3 = st.columns(3)
            b1.markdown(f"<div class='badge OK'>NPV: {npv_val:,.0f} k€</div>", unsafe_allow_html=True)
            b2.markdown(f"<div class='badge OK'>IRR: {irr_val*100:,.1f} %</div>", unsafe_allow_html=True)
            b3.markdown(f"<div class='badge OK'>Payback: {pb:.2f} ans</div>", unsafe_allow_html=True)
        st.markdown("<p class='small'>Exemples issus de cas Odoo (coûts réduits, modularité, TCO bas). Personnaliser avec vos chiffres réels.</p>", unsafe_allow_html=True)

        vals=bsc.loc[bsc["KPI"]=="ROI ERP (%)",["2023","2024","2025"]].iloc[0].values.tolist()
        st.plotly_chart(fig_roi_series([2023,2024,2025], vals, title="ROI ERP (%) — cible"), use_container_width=True, key="d1_roi")

    with s3:
        st.markdown("#### Gouvernance — acteurs & flux (survol = explications)")
        nodes=[
            {"id":"MOA","label":"MOA (DAF/Opérations)","title":"Priorise besoins, ROI, conformité","color":"#0ea5e9","size":node_size},
            {"id":"MOE","label":"MOE (DSI/Intégrateur)","title":"Paramétrage Odoo, sécurité, intégrations","color":"#22c55e","size":node_size},
            {"id":"OPS","label":"OPS/Prod","title":"Dispo, sauvegardes, patching, monitoring","color":"#eab308","size":node_size},
            {"id":"COM","label":"Comité SI (COPIL)","title":"Arbitre portefeuille, priorités, KPI, risques","color":"#ef4444","size":node_size+6},
            {"id":"ERP","label":"Odoo ERP","title":"CRM/Achats/Stock/RH/Compta (PostgreSQL)","color":"#6366f1","size":node_size+4,"shape":"ellipse"},
            {"id":"BI","label":"BI/DataMart","title":"OLAP, KPIs, réplication/ETL contrôlée","color":"#a855f7","size":node_size,"shape":"ellipse"},
            {"id":"EXT","label":"APIs externes","title":"Banque, e-commerce, TMS/WMS","color":"#f97316","size":node_size,"shape":"ellipse"}
        ]
        edges=[
            {"src":"MOA","dst":"COM","title":"Besoins / ROI"},
            {"src":"MOE","dst":"COM","title":"Chiffrage / Sécurité"},
            {"src":"COM","dst":"MOE","title":"Go/NoGo"},
            {"src":"MOE","dst":"ERP","title":"Paramétrage / Dev"},
            {"src":"OPS","dst":"ERP","title":"Ops / Sauvegardes / Patching"},
            {"src":"ERP","dst":"BI","title":"ETL / réplication"},
            {"src":"ERP","dst":"EXT","title":"REST / Webhooks"}
        ]
        pyvis_graph(nodes, edges, height="630px", physics=physics)

    with s4:
        st.markdown("#### Decision Rights (RACI)")
        df_raci = pd.DataFrame({
            "Décision":[
                "Évolution module Comptabilité","Ouverture API marketplace","Politique sauvegardes/DR",
                "Refonte rôles & SoD","Politique logs & rétention","Stratégie intégration (ESB/API)"
            ],
            "R":["MOE","MOE","OPS","MOE","OPS","MOE"],
            "A":["Comité SI"]*6,
            "C":["MOA, Auditeur","MOA, Sécu","MOA, Sécu","MOA, DPO","MOE, DPO","Architecte, Sécu"],
            "I":["Users clés"]*6
        })
        st.dataframe(df_raci, hide_index=True, use_container_width=True)
        st.markdown("<p class='small'>KPI de gouvernance : % changements planifiés, délai de résolution P1, couverture tests, dette technique.</p>", unsafe_allow_html=True)

    with s5:
        if show_sources:
            st.markdown("#### Sources & références (fr/en/de…)")
            for title, url in SOURCES["Jour 1 — Business Case & Gouvernance"]:
                st.markdown(f"- [{title}]({url})")

# ===========================================================================
# DAY 2
# ===========================================================================
with d2:
    p1, p2, p3, p4, p5 = st.tabs([
        "BPMN (O2C/P2P)",
        "Architecture logique (3-tiers)",
        "Architecture réseau (DMZ/Prod/Data)",
        "Intégrations & DataFlow",
        "Sources"
    ])

    with p1:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("##### O2C — Order to Cash (BPMN simplifié)")
            st.graphviz_chart(bpmn_o2c_graph(), use_container_width=True)
        with c2:
            st.markdown("##### P2P — Procure to Pay (BPMN simplifié)")
            st.graphviz_chart(bpmn_p2p_graph(), use_container_width=True)
        st.markdown("<p class='small'>Contrôles : 3-way match P2P ; validations montant ; relances & lettrage O2C ; traçabilité bout-en-bout.</p>", unsafe_allow_html=True)

    with p2:
        st.markdown("##### 3-tiers & intégrations (survol = détails)")
        nodes=[
            {"id":"FE","label":"Front (Web/Mobile)","title":"UI, sessions, i18n","color":"#22c55e","size":node_size+2},
            {"id":"WAF","label":"WAF/Reverse-Proxy","title":"TLS, WAF, rate-limit, caching","color":"#f97316","size":node_size+2},
            {"id":"ODOO","label":"Odoo Server","title":"Modules, ORM, bus, REST, workers","color":"#3b82f6","size":node_size+6},
            {"id":"PG","label":"PostgreSQL","title":"WAL, PITR, index, partitions","color":"#a855f7","size":node_size+4},
            {"id":"DWH","label":"DataMart/OLAP","title":"ETL contrôlé, SCD, historisation","color":"#eab308","size":node_size+2},
            {"id":"EXT","label":"APIs externes","title":"Banque, e-commerce, TMS/WMS","color":"#ef4444","size":node_size+2}
        ]
        edges=[
            {"src":"FE","dst":"WAF","title":"HTTPS/TLS 1.2+"},
            {"src":"WAF","dst":"ODOO","title":"Reverse proxy / headers sécurité"},
            {"src":"ODOO","dst":"PG","title":"ORM / SQL"},
            {"src":"ODOO","dst":"DWH","title":"ETL / réplication"},
            {"src":"ODOO","dst":"EXT","title":"REST/Webhooks"}
        ]
        pyvis_graph(nodes, edges, height="590px", physics=physics)

    with p3:
        st.markdown("##### Réseau segmenté (DMZ / App / Data) + sauvegardes immuables")
        nodes=[
            {"id":"USR","label":"Users (VPN+MFA)","title":"SSO/MFA, RBAC app","color":"#16a34a","size":node_size},
            {"id":"FW","label":"Firewall","title":"ACL, micro-segmentation","color":"#10b981","size":node_size},
            {"id":"DMZ","label":"DMZ/WAF","title":"Inspection, anti-bot","color":"#06b6d4","size":node_size},
            {"id":"APP","label":"Odoo App","title":"Autoscaling, patching","color":"#3b82f6","size":node_size+4},
            {"id":"DB","label":"PostgreSQL","title":"WAL, réplication, PITR","color":"#8b5cf6","size":node_size+2},
            {"id":"BKP","label":"Backups immuables","title":"Obj. storage, 30j, test restore","color":"#f59e0b","size":node_size},
            {"id":"SIEM","label":"SIEM/Logs","title":"Logs signés, alerting corrélé","color":"#ef4444","size":node_size}
        ]
        edges=[
            {"src":"USR","dst":"FW"},{"src":"FW","dst":"DMZ"},{"src":"DMZ","dst":"APP"},
            {"src":"APP","dst":"DB"},{"src":"APP","dst":"BKP","title":"Snapshots"},
            {"src":"DB","dst":"BKP","title":"WAL/PITR"},{"src":"APP","dst":"SIEM"},{"src":"DB","dst":"SIEM"}
        ]
        pyvis_graph(nodes, edges, height="560px", physics=physics)
        st.plotly_chart(fig_capacity_diurnal(), use_container_width=True, key="d2_load")

    with p4:
        st.markdown("##### Flux d’intégration (REST, Webhooks, ETL planifié)")
        nodes=[
            {"id":"EC","label":"E-commerce","title":" commandes web ","color":"#06b6d4","size":node_size},
            {"id":"OD","label":"Odoo","title":"API JSON/XML-RPC","color":"#3b82f6","size":node_size+4},
            {"id":"BK","label":"Banque/SEPA","title":"relevés, paiements","color":"#22c55e","size":node_size},
            {"id":"BI","label":"BI / DWH","title":"ETL horaire","color":"#a855f7","size":node_size},
            {"id":"WMS","label":"WMS/TMS","title":"logistique","color":"#ef4444","size":node_size}
        ]
        edges=[
            {"src":"EC","dst":"OD","title":"webhooks commande"},
            {"src":"OD","dst":"BK","title":"SEPA/fichiers"},
            {"src":"OD","dst":"BI","title":"ETL"},
            {"src":"WMS","dst":"OD","title":"stock/expé"}
        ]
        pyvis_graph(nodes, edges, height="520px", physics=physics)

    with p5:
        if show_sources:
            st.markdown("#### Sources & références")
            for title, url in SOURCES["Jour 2 — Processus & Architecture"]:
                st.markdown(f"- [{title}]({url})")

# ===========================================================================
# DAY 3
# ===========================================================================
with d3:
    q1, q2, q3, q4, q5 = st.tabs([
        "Data Governance & DQ",
        "Modèle étoile & ETL",
        "Sécurité (CIAN)",
        "SLA / Monitoring",
        "Sources"
    ])

    with q1:
        st.markdown("##### Règles Data Quality (exemples)")
        dq = pd.DataFrame({
            "ID":["DQ1","DQ2","DQ3","DQ4","DQ5","DQ6","DQ7"],
            "Règle":["Unicité client_id","Prix ≥ 0","TVA ∈ référentiel","Date ≤ today","IBAN format/clé","Devise ∈ ISO4217","Email valide"],
            "Severité":["High","High","Medium","Medium","High","Low","Low"],
            "Contrôle":["ETL pre-load","ETL pre-load","Lookup Dim_Taxes","ETL pre-load","Regex+mod97","Lookup table","Regex RFC5322"],
            "Action":["Reject","Reject","Warn","Warn","Reject","Warn","Warn"]
        })
        st.dataframe(dq, use_container_width=True, hide_index=True)
        st.markdown("<p class='small'>Mesures : % rejets, temps correction, #tickets DQ, couverture des règles, fraîcheur référentiels.</p>", unsafe_allow_html=True)

    with q2:
        st.markdown("##### Modèle étoile (ventes) + pipeline ETL contrôlé")
        c1,c2 = st.columns([1.2,1])
        with c1:
            st.code("""Fait_Ventes(date_id, client_id, article_id, qté, prix, montant)
Dim_Clients(client_id, segment, pays)
Dim_Articles(article_id, famille, tva)
Dim_Temps(date_id, mois, trimestre, année)
Clés: (client_id, article_id, date_id)""", language="sql")
        with c2:
            st.markdown("""**ETL (contrôlé)**
- Extraction: journaux Odoo (REST/DB/CSV)
- Contrôles: DQ rules (unicité, référentiels)
- Transformation: mapping TVA, normalisation
- Chargement: SCD, historisation, journalisation
- Audit: rejets, piste d’audit, réconciliation""")
        st.plotly_chart(fig_revenus_ebitda(), use_container_width=True, key="d3_rev")
        st.plotly_chart(fig_incidents(), use_container_width=True, key="d3_inc")

    with q3:
        st.markdown("##### Politique CIAN & contrôles techniques")
        st.markdown("""- **Confidentialité** : TLS1.2+, AES-256 at-rest, KMS (rotation 90j)
- **Intégrité** : hash ETL, contrôles de référence, signatures journaux
- **Authentification** : SSO/MFA, RBAC, SoD, JML automatisé
- **Non-répudiation** : horodatage, preuve d’origine (logs signés), audit trail""")
        st.markdown("<p class='small'>Tests d’intrusion annuels, scans de vulnérabilités mensuels, remédiation ≤ 30j (High).</p>", unsafe_allow_html=True)

    with q4:
        st.markdown("##### SLA, SLO & Error Budget")
        c1,c2,c3 = st.columns(3)
        with c1:
            target = st.slider("SLA cible (%)", 99.0, 99.99, 99.7)
        with c2:
            days = st.slider("Période (jours)", 7, 31, 30)
        with c3:
            budget = error_budget_minutes(target, days)
            st.markdown(f"<div class='badge WARN'>Error budget : {budget} min / {days} j</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_sla_bar(), use_container_width=True, key="d3_sla")
        st.plotly_chart(fig_latency_error(), use_container_width=True, key="d3_perf")
        st.markdown("<p class='small'>SLO modules critiques ≥99.7%. MTTR & MTTD suivis en temps réel. Journalisation centralisée (ELK / Loki / SIEM).</p>", unsafe_allow_html=True)

    with q5:
        if show_sources:
            st.markdown("#### Sources & références")
            for title, url in SOURCES["Jour 3 — Données, Cyber, SLA"]:
                st.markdown(f"- [{title}]({url})")

# ===========================================================================
# DAY 4
# ===========================================================================
with d4:
    r1, r2, r3, r4, r5 = st.tabs([
        "RCM priorisée & Heatmap",
        "Programme d’audit (COBIT/ISO)",
        "Reporting (BSC)",
        "Export PDF (rapport long)",
        "Sources"
    ])

    with r1:
        st.dataframe(rcm.sort_values("Score",ascending=False), use_container_width=True, hide_index=True)
        st.plotly_chart(fig_risk_heatmap(rcm), use_container_width=True, key="d4_heat")
        st.markdown("<p class='small'>Priorités: R5 Cyber (15), R3 Dispo (10), R1 Accès (12). Traiter avant go-live (contrôles, runbooks, tests).</p>", unsafe_allow_html=True)

    with r2:
        st.markdown("""#### Plan d’audit (extraits)
- **Portée** : Accès/SoD, Journalisation, Changements (CI/CD), Sauvegardes/DR, Intégrations/API, DQ/ETL
- **Tests** :
  1) Revue RBAC / comptes orphelins (échantillon utilisateurs + logs d’admins)  
  2) Traçabilité factures (échantillon 30) : du document source au GL  
  3) Restore à blanc (1/semestre) + RTO mesuré  
  4) Rejeu ETL + contrôles de cohérence (agrégats, totaux)  
  5) API : rate-limit, auth, input validation, logs d’erreurs  
  6) Conformité RGPD : registre traitements, purge, droits personnes
- **Référentiels** : COBIT 5/2019 (EDM, APO, BAI, DSS), ISO 27001 (A.5–A.18), ITIL (Change/Incident/Problem).
- **Évidence** : exports Odoo (CSV/PDF), logs signés, captures d’écran, scripts d’audit, rapports de tests.""")

    with r3:
        st.dataframe(bsc, use_container_width=True, hide_index=True)
        vals=bsc.loc[bsc["KPI"]=="ROI ERP (%)",["2023","2024","2025"]].iloc[0].values.tolist()
        st.plotly_chart(fig_roi_series([2023,2024,2025], vals, title="ROI ERP (%) — trajectoire"), use_container_width=True, key="d4_roi")
        st.markdown("<p class='small'>BSC : 4 perspectives — financière, client, processus, apprentissage. Piloter mensuellement (MoM/YoY).</p>", unsafe_allow_html=True)

    with r4:
        st.markdown("##### Export PDF complet (visuels + texte structuré)")
        if st.button("Générer le PDF (rapport complet)"):
            img1 = mpl_from_plotly(fig_revenus_ebitda(), "Revenus & EBITDA")
            img2 = mpl_bar_snapshot(fig_capex_opex(), "CAPEX vs OPEX")
            img3 = mpl_bar_snapshot(fig_incidents(), "Incidents IT/an")
            img4 = mpl_from_plotly(fig_latency_error(), "Perf P95 & erreurs")
            img5 = mpl_bar_snapshot(fig_sla_bar(), "SLA par module")
            img6 = mpl_from_plotly(fig_capacity_diurnal(), "Charge applicative (req/min)")
            img7 = mpl_from_plotly(fig_risk_heatmap(rcm), "Heatmap risques")
            sections = [
                {"title":"1) Contexte & objectifs",
                 "text":"Positionner Odoo comme SI cœur: base unique, intégration des processus, gouvernance claire. Objectifs: fiabilité, conformité, performance, valeur (ROI)."},
                {"title":"2) Business Case — Finance",
                 "text":"Gains: réduction erreurs, accélération O2C, discipline trésorerie. KPI: revenus, EBITDA, marge, DSO.",
                 "images":[(img1, 13.5), (img2, 13.5)]},
                {"title":"3) Incidents & Fiabilité",
                 "text":"Baisse continue des incidents; prioriser P1/P2; MTTR suivi.",
                 "images":[(img3, 13.5)]},
                {"title":"4) Perf applicative",
                 "text":"P95 < 400ms cible; error budget mensuel 0,5%.",
                 "images":[(img4, 13.5)]},
                {"title":"5) Processus intégrés (O2C/P2P)",
                 "text":"Contrôles de passage: validation commande, stock, facture, rapprochement; 3-way match achats."},
                {"title":"6) Architecture (logique/physique)",
                 "text":"3-tiers (Front→WAF→Odoo→PG); DMZ/WAF; sauvegardes immuables; SIEM; réplication DWH.",
                 "images":[(img5, 13.5), (img6, 13.5)]},
                {"title":"7) RCM — Risques & contrôles",
                 "text":"Priorités: Cyber (15), Dispo (10), Accès (12). Plans d’action & runbooks.",
                 "images":[(img7, 13.5)]},
                {"title":"8) Audit & Reporting",
                 "text":"Programme d’audit: accès/SoD, logs, changes, DR, API, ETL/DQ; référentiels COBIT/ISO/ITIL. BSC: 4 perspectives."}
            ]
            pdf = export_pdf("Odoo — SI (PhD) — Rapport complet", sections)
            st.download_button("⬇️ Télécharger le PDF",
                               data=pdf,
                               file_name=f"Odoo_SI_PhD_{datetime.now():%Y%m%d_%H%M}.pdf",
                               mime="application/pdf")

    with r5:
        if show_sources:
            st.markdown("#### Sources & références")
            for title, url in SOURCES["Jour 4 — Audit, Contrôles, Reporting"]:
                st.markdown(f"- [{title}]({url})")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("")
