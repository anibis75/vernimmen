# -*- coding: utf-8 -*-
# Odoo — Management & Audit des SI — v3.0 (3 onglets épurés)
# Day 1: Gouvernance | Day 2: Artefacts (Tableau + Création/Suppression) | Day 3: Risques + IA + SLA
# Lancer :  streamlit run app_odoo_si_phd.py

from __future__ import annotations
import streamlit as st, pandas as pd, numpy as np, requests, tempfile, os
from pathlib import Path
from datetime import datetime
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# CONFIG + THEME
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Odoo — SI ", page_icon="🧭", layout="wide")
st.markdown("""
<style>
:root { --bg:#0a0f1d; --panel:#0f172a; --muted:#a5b4c2; --border:#1e293b; --accent:#38bdf8; --fg:#e5e7eb; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui; }
h1,h2,h3 { margin: .2rem 0 .6rem 0; color: var(--fg); }
hr { height:1px; border:0; background: linear-gradient(90deg,var(--accent),transparent); margin: 8px 0 18px; }
.block { background: var(--panel); color: var(--fg); border-radius: 16px; padding: 16px 18px; border: 1px solid var(--border); }
.small { color: var(--muted); font-size: 0.92rem; }
.badge { display:inline-flex; align-items:center; gap:.4rem; padding:.15rem .55rem; border-radius:10px; border:1px solid var(--border); background:#0c1426; color:#cfd6df; margin:.15rem .3rem .15rem 0; }
.badge.OK{border-color:#1d3b2a;color:#a7f3d0;background:#052e1a;}
.badge.WARN{border-color:#3b2d05;color:#fde68a;background:#1f1501;}
.badge.KO{border-color:#4b1111;color:#fecaca;background:#1f0a0a;}
.stTabs [data-baseweb="tab-list"] { gap: 10px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: #0c1426; border-radius: 12px; padding: 10px 16px; border: 1px solid var(--border); color:#dbe2ea; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { border-color: var(--accent); box-shadow: inset 0 -3px 0 var(--accent); color: #fff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DEMO DATA (SLA/Perf)
# ---------------------------------------------------------------------------
np.random.seed(42)
YEARS = np.array([2021,2022,2023,2024,2025,2026])
lat_p95 = np.array([520,470,430,390,360,340])
err_rate = np.array([0.80,0.70,0.60,0.45,0.35,0.30])

modules = ["ERP Core","Comptabilité","CRM","Achats","Stock","RH","Projet","E-commerce"]
sla = np.clip(np.random.normal(99.74,0.18,len(modules)), 99.0, 99.95).round(2)
df_sla = pd.DataFrame({"Module":modules,"SLA (%)":sla}).sort_values("SLA (%)", ascending=False)

# --- BUSINESS DATA (offres, pricing, funnel, modules, régions) ---
REGIONS = ["UE", "USA", "MEA", "APAC"]
SERVICES = ["Implémentation", "Support", "Formation", "Dév. spécifique", "Hébergement Odoo.sh"]

PRICING = pd.DataFrame([
    # Offre, Type, Prix mensuel (€/user), Cible
    ["Odoo Online (1 app)", "Freemium", 0, "TPE / test"],
    ["Odoo Standard", "SaaS", 19, "TPE/PME"],
    ["Odoo Custom", "SaaS", 29, "PME/ETI"],
    ["Odoo On-prem", "Licence/Entretien", 39, "ETI/GE"],
], columns=["Offre","Type","Prix (€/user/mois)","Cible"])

MODULES = pd.DataFrame({
    "Module":[
        "Comptabilité","Ventes","Achat","Stock","Fabrication (MRP)","Projet","RH/Paie",
        "CRM","Marketing (Email/SMS)","E-commerce","Point de Vente (PoS)","Helpdesk"
    ],
    "Maturité":["Très élevé","Élevé","Élevé","Élevé","Moyen+","Élevé","Moyen",
                "Élevé","Moyen","Élevé","Élevé","Moyen"]
})

# Funnel acquisition → activation → rétention (valeurs démo)
FUNNEL = pd.DataFrame({
    "Étape":["Visites","Essais (sign-up)","Activation (D+14)","Payants (M1)","Rétention (M6)"],
    "Taux conv. (%)":[100, 6.5, 3.2, 1.8, 1.2]
})
FUNNEL["Absolu (base=100k)"] = (FUNNEL["Taux conv. (%)"]/100*100_000).round(0).astype(int)

# CAC/LTV par région (démo) — € par client
UNIT_ECO = pd.DataFrame({
    "Région":REGIONS,
    "CAC (€/client)":[190, 240, 130, 170],
    "ARPA (€/mois)":[38, 42, 28, 33],
    "Churn mensuel (%)":[2.6, 2.3, 3.2, 2.9],
})
UNIT_ECO["LTV (€/client)"] = (UNIT_ECO["ARPA (€/mois)"] / (UNIT_ECO["Churn mensuel (%)"]/100) * 0.7).round(0)
UNIT_ECO["LTV/CAC"] = (UNIT_ECO["LTV (€/client)"]/UNIT_ECO["CAC (€/client)"]).round(2)

# Mix de revenus par service (démo, %)
REV_MIX = pd.DataFrame({
    "Service":SERVICES,
    "Part (%)":[32, 18, 12, 23, 15]
})

# -------- Figures --------
def fig_funnel(funnel: pd.DataFrame):
    f = go.Figure(go.Funnel(
        y=funnel["Étape"],
        x=funnel["Absolu (base=100k)"],
        text=[f"{v:.1f}%" for v in funnel["Taux conv. (%)"]],
        textposition="inside"
    ))
    f.update_layout(height=420, template="plotly_white", title="Funnel acquisition → activation → rétention (base: 100k visites)")
    return f

def fig_rev_mix(df: pd.DataFrame):
    f = go.Figure(go.Bar(x=df["Service"], y=df["Part (%)"], text=df["Part (%)"], textposition="auto"))
    f.update_layout(height=360, template="plotly_white", title="Mix de revenus par service (%)", yaxis_range=[0, max(df["Part (%)"])+10])
    return f

def fig_unit_heat(df: pd.DataFrame):
    h = df[["CAC (€/client)","LTV (€/client)","LTV/CAC"]].copy()
    h.index = df["Région"]
    f = px.imshow(h.T, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                  title="Unit Economics — CAC, LTV, LTV/CAC (par région)")
    f.update_layout(height=360, template="plotly_white")
    return f


# ---------------------------------------------------------------------------
# IA — Matrice d'opportunités (concret Odoo)
# ---------------------------------------------------------------------------
AI_OPPS = pd.DataFrame([
    ["AI1","Prévision des ventes (SKU)","Ventes/Stock","sale.order_line, stock.move, saisonnalité, promos","Prophet/ARIMA XGB",45,10,180,5,3,3,"DAF/DSI"],
    ["AI2","Lead scoring (conversion)","CRM","crm.lead, source, campagne, interactions mail","LogReg/XGB",20,6,60,3,2,2,"Ventes/Marketing"],
    ["AI3","Détection anomalies factures","Comptabilité","account.move(lines), tiers, historiques","IsolationForest/Autoencoder",30,8,120,4,3,4,"DAF/Contrôle int."],
    ["AI4","Optimisation réassort","Achats/Stock","stock.quant, lead time, prix, ruptures","RL/Prog linéaire",55,12,160,5,4,5,"Achats/Logistique"],
    ["AI5","Chat d’assistance interne","Support/Docs","mail.message, helpdesk.ticket, docs","RAG + LLM",25,9,70,3,2,2,"DSI/Support"],
    ["AI6","Prévision retards paiement","Compta (AR)","account.move, terms, historiques règlements","XGB + survie",25,6,110,4,2,3,"DAF/Trésorerie"],
    ["AI7","Réconciliation auto enrichie","Banque","account.bank.statement, rules","Similarity + règles ML",18,5,55,3,2,2,"DAF/Trésorerie"],
    ["AI8","Prix dynamiques & stock","E-commerce/Stock","prix concurrents, stock, élasticité","Bandits contextuels",60,15,190,5,4,5,"E-comm/Marketing"],
    ["AI9","Prédiction churn clients","CRM/Ventes","RFM, tickets, commandes","XGB/LogReg + SHAP",28,7,90,4,3,3,"Ventes/CS"],
    ["AI10","Classement tickets auto","Helpdesk","helpdesk.ticket, tags, historiques","BERT small / distil",22,6,65,3,2,2,"Support/DSI"],
], columns=["ID","Use case","Module","Données","Modèle","CAPEX (k€)","OPEX/an (k€)","Gains/an (k€)","Impact","Complexité","TTV (mois)","Owner"])
AI_OPPS["ROI 12m (k€)"] = AI_OPPS["Gains/an (k€)"] - AI_OPPS["OPEX/an (k€)"] - (AI_OPPS["CAPEX (k€)"]/12)*AI_OPPS["TTV (mois)"]
AI_OPPS["Score priorité"] = (AI_OPPS["Impact"]*2 + np.clip(6-AI_OPPS["Complexité"],1,5)) * (AI_OPPS["ROI 12m (k€)"]>0).astype(int)

def fig_ai_matrix(df: pd.DataFrame):
    f = px.scatter(
        df, x="Complexité", y="Impact", size="ROI 12m (k€)", color="Module",
        hover_data=["ID","Use case","Gains/an (k€)","CAPEX (k€)","OPEX/an (k€)","TTV (mois)","Owner"],
        size_max=60, template="plotly_white", title="Matrice IA — Impact vs Complexité (taille = ROI 12m)"
    )
    f.update_layout(height=520, xaxis=dict(dtick=1, range=[0.5,5.5]), yaxis=dict(dtick=1, range=[0.5,5.5]))
    return f

# ---------------------------------------------------------------------------
# CYBER — Risques + simulateur
# ---------------------------------------------------------------------------
CYBER_RISKS = pd.DataFrame([
    ["C1","Accès","Credential stuffing /web/login","Bots+listes",4,4,"MFA/SSO, rate-limit, CAPTCHA, IP allow", "DSI/Sécu"],
    ["C2","Accès","Escalade via ACL","Param rôles",3,5,"RBAC, SoD, recertif, revues logs", "DSI/MOE"],
    ["C3","App","Injection (module custom)","Entrées non filtrées",3,5,"Validation, revues code, SAST/DAST", "MOE/Qualité"],
    ["C7","Infra","Ransomware VM Odoo/PG","Phishing/EDR contourné",3,5,"EDR, backups immuables, drills", "SOC/Infra"],
    ["C10","Secrets","Clés dans repo","Erreur dev/CI",3,4,"Vault/KMS, scans secrets, rotation", "MOE/Sécu"],
    ["C11","Intégrations","API non auth ou clé exposée","Lien partenaire",3,4,"OAuth2/HMAC, scopes, quotas", "DSI/MOE"],
    ["C12","Données","Bucket S3/MinIO public","ACL publique",3,5,"Block public ACLs, SSE-KMS", "Infra/Sécu"],
    ["C15","Ops","Backups non testés / MTTR haut","DR non testé",3,5,"PITR, tests réguliers, runbooks", "Infra/DSI"],
], columns=["ID","Domaine","Risque","Vecteur","Probabilité","Impact","Contrôle clé","Owner"])
CYBER_RISKS["Score"] = CYBER_RISKS["Probabilité"] * CYBER_RISKS["Impact"]

def fig_cyber_heatmap(df: pd.DataFrame):
    grid = pd.DataFrame(0, index=[1,2,3,4,5], columns=[1,2,3,4,5])
    for _, r in df.iterrows():
        grid.loc[r["Impact"], r["Probabilité"]] += 1
    grid = grid.sort_index(ascending=True)
    f = px.imshow(grid, text_auto=True, color_continuous_scale="Reds", aspect="auto")
    f.update_layout(height=420, template="plotly_white",
                    title="Heatmap Cyber (nb par case Impact×Probabilité)",
                    xaxis_title="Probabilité", yaxis_title="Impact")
    return f

def error_budget_minutes(sla_target: float, days: int=30) -> int:
    minutes = days*24*60
    return int(round((1 - sla_target/100.0) * minutes))

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

# ---------------------------------------------------------------------------
# ARTEFACTS — images distantes/locales
# ---------------------------------------------------------------------------
# Image CEO (Fabien Pinckaers)
CEO_IMG_CANDIDATES = [
    "https://rtleng.rosselcdn.net/sites/default/files/dpistyles_v2/ena_16_9_extra_big/2023/06/28/node_563836/2918131/public/2023/06/28/13794232.jpg?itok=SN1grTkf1687945662",
    r"C:\Users\azad1\Documents\Skema\SI\odoo_ceo.png",   # option locale (facultative)
    "/mnt/data/odoo_ceo.png",                            # option cloud (facultative)
]

CREATION_IMG_CANDIDATES = [
    "https://anibis75.github.io/vernimmen/Creation%20de%20compte.png",
    r"C:\Users\azad1\Documents\Skema\SI\Creation de compte.png",
    "/mnt/data/Creation de compte.png",
]
TABLEAU_IMG_CANDIDATES = [
    "https://anibis75.github.io/vernimmen/tableau.png",
    r"C:\Users\azad1\Documents\Skema\SI\tableau.png",
    "/mnt/data/tableau.png",
]
SUPPRESSION_IMG_CANDIDATES = [
    "https://anibis75.github.io/vernimmen/suppression.png",
    r"C:\Users\azad1\Documents\Skema\SI\suppression.png",
    "/mnt/data/suppression.png",
]

# Image de référence concurrence (évite /mnt/data par défaut)
COMPETITION_IMG_CANDIDATES = [
    "https://anibis75.github.io/vernimmen/frontrunners_erp.png",
    r"C:\Users\azad1\Documents\Skema\SI\frontrunners_erp.png",
]

def is_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))

def first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if not p: continue
        if is_url(p): return p
        if Path(p).exists(): return p
    return None

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("<h1>🧭 Odoo — Management & Audit des SI</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3 ONGLETS PRINCIPAUX
# ---------------------------------------------------------------------------
d1, d2, d3 = st.tabs([
    "Day 1 — Gouvernance du Système d’Information — Odoo",
    "Day 2 — Artefacts (Tableau + Création/Suppression)",
    "Day 3 — Risques, Data (IA) & SLA"
])

# ===================== Day 1 — GOUVERNANCE =====================
with d1:
    # 3 sous-onglets
    g2, g1, g3 = st.tabs(["Business Case", "Governance", "Analyse concurrence"])

    # ---- Sous-onglet 1 : Governance ----
    with g1:
        st.markdown("### Gouvernance du Système d’Information — Odoo")
        st.markdown("""
**Pilotes & rôles.** Binôme produit–tech : **Fabien Pinckaers** (CEO) oriente le produit; **Antony Lesuisse** (CTO) arbitre architecture/tech.
Fonctions clés : **CISO**, **DPO**, **Head of Infrastructure** (SRE/ops). Gouvernance 3 lignes : run / risque-conformité / audit.

**Comités & décisions.** **Comité SI** mensuel, **CAB** (Go/NoGo), comité **Sécurité & Conformité** (OWASP/CVE, ISO 27001/27701, SOC 2, RGPD).

**Cadre & pratiques.** DevSecOps CI/CD, environnements dev/stage/prod, segmentation DMZ/WAF → App → Data, backups immuables (RTO/RPO), observabilité.
Référentiels : ITIL 4, COBIT, TOGAF/IAF, ISO 27001/27701.

**Droits & traçabilité.** RBAC + SoD, JML, recertifications périodiques, logs signés/horodatés, opérations 4-yeux, API/webhooks filtrés.

**Indicateurs.** Dispo ≥ 99,9 %, MTTR < 2 h, SLO P95 < 400 ms, coverage tests ≥ 80–85 %, release annuelle + patch mensuel.
Tableau de bord : SLA, incidents P1/P2, dette technique, vulnérabilités, conformité, coûts (CAPEX/OPEX).
        """)

    # ---- Sous-onglet 2 : Business Case ----
    with g2:
        st.markdown("### Business Case — Odoo (marché, offres, go-to-market, unit economics)")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown("<div class='badge OK'>Positionnement: ERP modulaire (PME→ETI)</div>", unsafe_allow_html=True)
        c2.markdown("<div class='badge OK'>Modèle: SaaS + Open Core</div>", unsafe_allow_html=True)
        c3.markdown("<div class='badge WARN'>Freemium: 1 app gratuite</div>", unsafe_allow_html=True)
        c4.markdown("<div class='badge OK'>Canaux: SEO/Content + Partenaires</div>", unsafe_allow_html=True)

        st.markdown("#### 1) Offres & pricing (démo)")
        st.dataframe(PRICING, use_container_width=True, hide_index=True)

        st.markdown("#### 2) Modules clés & maturité")
        st.dataframe(MODULES, use_container_width=True, hide_index=True)

        st.markdown("#### 3) Go-to-Market — acquisition → activation → rétention")
        st.plotly_chart(fig_funnel(FUNNEL), use_container_width=True, key="funnel_d1")
        st.markdown("""
- **Acquisition** : SEO produit, docs, tutoriels, communauté, marketplace apps, événements.
- **Activation** : essai guidé, data import wizard, modèles compta locaux, e-commerce prêt à l’emploi.
- **Rétention / Expansion** : cross-sell modules (ex. CRM→Ventes→Compta), intégrations, Odoo.sh, support & formations.
        """)

        st.markdown("#### 4) Freemium & Open Core")
        st.markdown("""
- **Freemium (1 app gratuite)** : abaisse le coût d’essai et accélère le time-to-value.
- **Open Core** : core et nombreux modules **open source**; add-ons/outilings & services premium (SaaS, Odoo.sh, support).
- **Monétisation** : abonnements par user, hébergement managé, services pro (implémentation, formation), marketplace.
        """)

        st.markdown("#### 5) Unit economics (CAC / LTV) — par région")
        st.dataframe(UNIT_ECO, use_container_width=True, hide_index=True)
        st.plotly_chart(fig_unit_heat(UNIT_ECO), use_container_width=True, key="unitheat_d1")

        st.markdown("#### 6) Mix de revenus par service")
        st.plotly_chart(fig_rev_mix(REV_MIX), use_container_width=True, key="revmix_d1")

        st.markdown("### 🌍 Data residency & hébergement cloud — Odoo (officiel 2025)")
        DC = pd.DataFrame([
            ["Saint-Ghislain 🇧🇪", 50.47, 4.11, "Europe (UE)", "Google Cloud / OVHcloud", "Odoo.sh + Odoo Online (EU)"],
            ["Iowa 🇺🇸", 41.88, -93.09, "Amériques (US)", "Google Cloud", "Odoo.sh + Odoo Online (US/CA)"],
            ["Dammam 🇸🇦", 26.43, 50.10, "Moyen-Orient", "Google Cloud", "Odoo.sh + Odoo Online (MEA)"],
            ["Mumbai 🇮🇳", 19.08, 72.88, "Asie du Sud", "Google Cloud", "Odoo.sh + Odoo Online (IN/SA)"],
            ["Singapore 🇸🇬", 1.35, 103.82, "Asie du Sud-Est", "Google Cloud", "Odoo.sh + Odoo Online (APAC)"],
            ["Sydney 🇦🇺", -33.86, 151.21, "Océanie", "Google Cloud", "Odoo.sh + Odoo Online (AU/NZ)"],
        ], columns=["Site","Latitude","Longitude","Région","Provider","Usage principal"])
        fig_map = px.scatter_geo(
            DC, lat="Latitude", lon="Longitude", text="Site", hover_name="Région",
            hover_data={"Provider":True, "Usage principal":True, "Latitude":False, "Longitude":False},
            size=[15]*len(DC), projection="natural earth", template="plotly_dark", color="Région",
            title="Localisation officielle des datacenters Odoo (Privacy Policy 2025)"
        )
        fig_map.update_traces(marker=dict(line=dict(width=1, color="#e5e7eb")))
        fig_map.update_layout(
            geo=dict(showland=True, landcolor="#1e293b", showocean=True, oceancolor="#0a192f",
                     showcountries=True, countrycolor="#555", showframe=False),
            height=520, margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                        font=dict(size=11, color="white"))
        )
        st.plotly_chart(fig_map, use_container_width=True, key="map_d1_g2")
        st.markdown("""
<div class='small'>
📄 <b>Sources officielles :</b><br>
• <a href="https://www.odoo.com/privacy" target="_blank">Odoo Privacy Policy – Data Location</a><br>
• <a href="https://www.odoo.sh/faq" target="_blank">Odoo.sh FAQ – Hosting zones</a>
</div>""", unsafe_allow_html=True)

        st.markdown("#### 7) Photo CEO (Fabien Pinckaers)")
        ceo_img = first_existing(CEO_IMG_CANDIDATES)
        up_ceo = st.file_uploader("", type=["png","jpg","jpeg"], key="up_ceo_biz")
        if up_ceo is not None:
            tmp = Path(tempfile.gettempdir()) / f"ceo_{datetime.now().timestamp():.0f}.png"
            with open(tmp, "wb") as f: f.write(up_ceo.read())
            ceo_img = str(tmp)
        if ceo_img:
            st.image(ceo_img, width=240, caption=Path(ceo_img).name)

    # ---- Sous-onglet 3 : Analyse concurrence ----
    with g3:
        st.markdown("### 🔎 Analyse concurrence — ERP PME/ETI (mapping interactif)")
        st.caption("Référence visuelle + mapping dynamique Usability × User-Recommended. Ajuste les pondérations pour le score composite.")

        # 1) Image de référence robuste (plus de /mnt/data par défaut)
        img_path = first_existing(COMPETITION_IMG_CANDIDATES)
        up_img = st.file_uploader("Remplacer l’image (optionnel)", type=["png","jpg","jpeg"], key="up_comp_img")
        if up_img is not None:
            tmpi = Path(tempfile.gettempdir()) / f"competition_{datetime.now().timestamp():.0f}.png"
            with open(tmpi, "wb") as f: f.write(up_img.read())
            img_path = str(tmpi)
        if img_path:
            st.image(img_path, caption="FrontRunners® ERP (réf. visuelle)", use_container_width=True)
        else:
            st.info("Image non trouvée. Dépose un PNG/JPG ci-dessus ou mets à jour COMPETITION_IMG_CANDIDATES.")

        # 2) Données démo (usability/recommended ~ échelle 3.2→4.8)
        data = pd.DataFrame([
            # Nom, Usability, Recommended, ARPA€/mois, Modules, OpenSource(1/0), CloudMaturity(1–5), Part marché PME (%)
            ["Odoo",                    3.95, 4.05, 29,  45, 1, 5, 8.5],
            ["ERPNext",                 4.10, 4.20,  0,  35, 1, 4, 2.1],
            ["Microsoft Dynamics 365",  4.40, 4.10, 65,  60, 0, 5, 12.4],
            ["NetSuite",                4.15, 3.95, 79,  55, 0, 5, 6.8],
            ["SAP Business One",        3.85, 3.75, 59,  50, 0, 4, 5.9],
            ["Priority ERP",            4.05, 4.25, 45,  40, 0, 4, 1.7],
        ], columns=["Vendor","Usability","Recommended","ARPA","Modules","OpenSrc","CloudM","SharePME"])

        # 3) Pondérations & filtres
        c1,c2,c3,c4 = st.columns([1,1,1,1])
        with c1:
            w_u = st.slider("Poids Usability", 0.0, 2.0, 1.0, 0.1)
        with c2:
            w_r = st.slider("Poids Recommended", 0.0, 2.0, 1.0, 0.1)
        with c3:
            w_c = st.slider("Poids Cloud Maturity", 0.0, 2.0, 0.6, 0.1)
        with c4:
            only_open = st.toggle("Filtrer Open Source", value=False)

        dfc = data.copy()
        if only_open:
            dfc = dfc[dfc["OpenSrc"] == 1]

        # 4) Score composite (0–100) + classement
        dfc["Score"] = (
            w_u * (dfc["Usability"] - 3.2)/(4.8-3.2) +
            w_r * (dfc["Recommended"] - 3.2)/(4.8-3.2) +
            w_c * (dfc["CloudM"] - 1)/(5-1)
        ) / max(w_u + w_r + w_c, 1e-6) * 100
        dfc = dfc.sort_values("Score", ascending=False)

        # 5) Mapping interactif
        fig_comp = px.scatter(
            dfc, x="Usability", y="Recommended", size="SharePME",
            color=dfc["OpenSrc"].map({1:"Open-source",0:"Propriétaire"}),
            hover_data=["Vendor","ARPA","Modules","CloudM","Score"],
            text="Vendor", size_max=60, template="plotly_white",
            title="Positionnement concurrentiel — Usability × User-Recommended"
        )
        fig_comp.update_traces(textposition="top center")
        fig_comp.update_layout(
            height=540, xaxis=dict(range=[3.2,4.8]), yaxis=dict(range=[3.2,4.8]),
            legend_title="Modèle", margin=dict(l=0,r=0,t=60,b=0)
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="comp_scatter_d1_g3")

# ---- Top 5 sans images ----
        st.markdown("#### 🏆 Top 5 (score composite)")

        top5 = dfc.head(5).reset_index(drop=True)
        cols = st.columns(5)

        for i, row in top5.iterrows():
            with cols[i]:
                st.markdown(
                    f"""
                    <div style='text-align:center; font-weight:600; color:#cbd5e1; font-size:1.1rem; margin-bottom:4px;'>
                        {row['Vendor']}
                    </div>
                    <div style='font-weight:bold; background:#013220; color:white; border-radius:8px; display:inline-block; padding:4px 10px;'>
                        Score {row['Score']:.1f}
                    </div>
                    <div style='font-size:0.9rem; color:#94a3b8; margin-top:4px;'>
                        ARPA: {row['ARPA']} €/mois • Modules: {row['Modules']} • CloudM: {row['CloudM']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # 7) Tableau comparatif + export
        st.markdown("#### Tableau comparatif")
        st.dataframe(
            dfc[["Vendor","Usability","Recommended","ARPA","Modules","OpenSrc","CloudM","SharePME","Score"]],
            use_container_width=True, hide_index=True
        )
        st.download_button("⬇️ Export CSV — Concurrence",
                           data=dfc.to_csv(index=False).encode("utf-8"),
                           file_name="odoo_competition_mapping.csv",
                           mime="text/csv")



        # ===== Carte data residency (duplicat, clé différente) =====
        st.markdown("### 🌍 Data residency & hébergement cloud — Odoo (officiel 2025)")
        DC = pd.DataFrame([
            ["Saint-Ghislain 🇧🇪", 50.47, 4.11, "Europe (UE)", "Google Cloud / OVHcloud", "Odoo.sh + Odoo Online (EU)"],
            ["Iowa 🇺🇸", 41.88, -93.09, "Amériques (US)", "Google Cloud", "Odoo.sh + Odoo Online (US/CA)"],
            ["Dammam 🇸🇦", 26.43, 50.10, "Moyen-Orient", "Google Cloud", "Odoo.sh + Odoo Online (MEA)"],
            ["Mumbai 🇮🇳", 19.08, 72.88, "Asie du Sud", "Google Cloud", "Odoo.sh + Odoo Online (IN/SA)"],
            ["Singapore 🇸🇬", 1.35, 103.82, "Asie du Sud-Est", "Google Cloud", "Odoo.sh + Odoo Online (APAC)"],
            ["Sydney 🇦🇺", -33.86, 151.21, "Océanie", "Google Cloud", "Odoo.sh + Odoo Online (AU/NZ)"],
        ], columns=["Site","Latitude","Longitude","Région","Provider","Usage principal"])

        fig_map = px.scatter_geo(
            DC, lat="Latitude", lon="Longitude", text="Site",
            hover_name="Région",
            hover_data={"Provider":True, "Usage principal":True, "Latitude":False, "Longitude":False},
            size=[15]*len(DC),
            projection="natural earth",
            template="plotly_dark",
            color="Région",
            title="Localisation officielle des datacenters Odoo (Privacy Policy 2025)"
        )
        fig_map.update_traces(marker=dict(line=dict(width=1, color="#e5e7eb")))
        fig_map.update_layout(
            geo=dict(showland=True, landcolor="#1e293b", showocean=True, oceancolor="#0a192f",
                     showcountries=True, countrycolor="#555", showframe=False),
            height=520, margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                        font=dict(size=11, color="white"))
        )
        st.plotly_chart(fig_map, use_container_width=True, key="map_d1_g3")

        st.markdown("""
<div class='small'>
📄 <b>Sources officielles :</b><br>
• <a href="https://www.odoo.com/privacy" target="_blank">Odoo Privacy Policy – Data Location</a><br>
• <a href="https://www.odoo.sh/faq" target="_blank">Odoo.sh FAQ – Hosting zones</a><br><br>
Les instances <b>Odoo Online</b> et <b>Odoo.sh</b> sont hébergées sur <b>Google Cloud Platform</b>, 
avec redondance <b>OVHcloud</b> pour certaines zones européennes (Belgique / France).<br>
Mise à jour confirmée : <i>septembre 2025</i>.
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 7) Photo CEO (Fabien Pinckaers)")
        ceo_img = first_existing(CEO_IMG_CANDIDATES)
        up_ceo = st.file_uploader("", type=["png","jpg","jpeg"], key="up_ceo_biz_g3")
        if up_ceo is not None:
            tmp = Path(tempfile.gettempdir()) / f"ceo_{datetime.now().timestamp():.0f}.png"
            with open(tmp, "wb") as f: f.write(up_ceo.read())
            ceo_img = str(tmp)
        if ceo_img:
            st.image(ceo_img, width=240, caption=Path(ceo_img).name)

# ===================== Day 2 — ARTEFACTS =====================
with d2:
    t1, t2, t3 = st.tabs([
        "Tableau ‘Architecture • Gouvernance • Sécurité’",
        "Création de compte (PNG)",
        "Suppression de compte (PNG)"
    ])
    with t1:
        img_path = first_existing(TABLEAU_IMG_CANDIDATES)
        up = st.file_uploader("Dépose l’image (optionnel)", type=["png","jpg","jpeg"], key="up_t1")
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"tableau_{datetime.now().timestamp():.0f}.png"
            with open(tmp, "wb") as f: f.write(up.read())
            img_path = str(tmp)
        if img_path: st.image(img_path, use_column_width=True, caption=Path(img_path).name)
        else: st.warning("Image non trouvée.")
        st.markdown("<p class='small'>Repère : WHY/WHAT/HOW/WITH × Domaines (Business, Information, IS, Infra, Governance, Security).</p>", unsafe_allow_html=True)

    with t2:
        img_path = first_existing(CREATION_IMG_CANDIDATES)
        up = st.file_uploader("Dépose l’image (optionnel)", type=["png","jpg","jpeg"], key="up_t2")
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"creation_{datetime.now().timestamp():.0f}.png"
            with open(tmp, "wb") as f: f.write(up.read())
            img_path = str(tmp)
        if img_path: st.image(img_path, use_column_width=True, caption=Path(img_path).name)
        else: st.warning("Image non trouvée.")
        st.markdown("**Contrôles clés** : reCAPTCHA, vérif MX, lien d’activation signé, time-box, double opt-in, anti-bruteforce, journaux signés.")

    with t3:
        img_path = first_existing(SUPPRESSION_IMG_CANDIDATES)
        up = st.file_uploader("Dépose l’image (optionnel)", type=["png","jpg","jpeg"], key="up_t3")
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"supp_{datetime.now().timestamp():.0f}.png"
            with open(tmp, "wb") as f: f.write(up.read())
            img_path = str(tmp)
        if img_path: st.image(img_path, use_column_width=True, caption=Path(img_path).name)
        else: st.warning("Image non trouvée.")
        st.markdown("**RGPD** : droit à l’effacement (sous réserve légale), pseudonymisation des traces, purge tokens/sessions, rupture consentements, sortie JML.")

# ===================== Day 3 — RISQUES + DATA (IA) + SLA =====================
with d3:
    q_risk, q_ai, q_sla = st.tabs([
        "Risques — Simulateur",
        "Data — Matrice IA",
        "SLA & Performance"
    ])

    # ---- Simulateur de risques
    with q_risk:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            attack = st.selectbox("Type d'attaque", [
                "Ransomware (poste/serveur)",
                "Credential stuffing (login Odoo)",
                "Injection (module custom)",
                "SSRF (webhook)",
                "Clé/secret exposé",
                "DoS via webhooks"
            ])
            waf = st.toggle("WAF actif", value=True)
        with c2:
            mfa = st.slider("Taux MFA/SSO (%)", 0, 100, 60)
            edr = st.slider("Couverture EDR (%)", 0, 100, 70)
        with c3:
            patch = st.slider("Latence patch (jours)", 0, 60, 14)
            backup_imm = st.toggle("Backups immuables + air-gap", value=True)
        with c4:
            rto = st.slider("RTO cible (heures)", 1, 72, 8)
            rpo = st.slider("RPO cible (minutes)", 0, 1440, 60)

        base = CYBER_RISKS["Score"].mean()
        m = 1.0
        if attack == "Credential stuffing (login Odoo)":
            m *= (1 - mfa/150) * (0.85 if waf else 1.0)
        elif attack == "Ransomware (poste/serveur)":
            m *= (1 - edr/120) * (0.7 if backup_imm else 1.2)
        elif attack == "Injection (module custom)":
            m *= (1 + patch/120)
        elif attack == "SSRF (webhook)":
            m *= (0.85 if waf else 1.1) * (1 + patch/200)
        elif attack == "Clé/secret exposé":
            m *= 1.25 * (1 - min(edr, mfa)/300)
        elif attack == "DoS via webhooks":
            m *= (0.9 if waf else 1.2)

        est_loss = base * 25_000 * m
        mttr_h = max(2, rto * (1.2 if not backup_imm else 0.8))
        avail = max(97.0, 100.0 - (mttr_h/24)*1.2)

        b1,b2,b3 = st.columns(3)
        b1.markdown(f"<div class='badge KO'>Perte attendue: {est_loss:,.0f} €</div>", unsafe_allow_html=True)
        b2.markdown(f"<div class='badge WARN'>MTTR estimé: {mttr_h:.1f} h</div>", unsafe_allow_html=True)
        b3.markdown(f"<div class='badge OK'>Dispo estimée: {avail:.2f} %</div>", unsafe_allow_html=True)

        st.plotly_chart(fig_cyber_heatmap(CYBER_RISKS), use_container_width=True, key="heatmap_risks_d3")
        st.markdown("##### Top risques (par score)")
        st.dataframe(CYBER_RISKS.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export CSV — RCM Cyber",
                           data=CYBER_RISKS.to_csv(index=False).encode("utf-8"),
                           file_name="rcm_cyber_odoo.csv", mime="text/csv")

    # ---- Data — IA
    with q_ai:
        st.markdown("### IA — Matrice d’opportunités (Odoo)")
        c1, c2, c3 = st.columns([1.2,1,1])
        with c1:
            mod_sel = st.multiselect("Filtrer par module", sorted(AI_OPPS["Module"].unique()),
                                     default=sorted(AI_OPPS["Module"].unique()))
        with c2:
            min_roi = st.number_input("ROI 12m minimal (k€)", value=0, step=10)
        with c3:
            max_ttv = st.slider("Time-to-Value (mois) max", 1, 12, 6)

        _df = AI_OPPS.query("`Module` in @mod_sel").copy()
        _df = _df[(_df["ROI 12m (k€)"] >= min_roi) & (_df["TTV (mois)"] <= max_ttv)]
        _df = _df.sort_values(["Score priorité","ROI 12m (k€)"], ascending=[False,False])

        st.plotly_chart(fig_ai_matrix(_df), use_container_width=True, key="ai_matrix_d3")
        st.markdown("#### Backlog priorisé")
        st.dataframe(
            _df[["ID","Use case","Module","Données","Modèle","CAPEX (k€)","OPEX/an (k€)","Gains/an (k€)","ROI 12m (k€)","Impact","Complexité","TTV (mois)","Owner","Score priorité"]],
            use_container_width=True, hide_index=True
        )
        st.download_button("⬇️ Export CSV — IA (backlog)",
                           data=_df.to_csv(index=False).encode("utf-8"),
                           file_name="odoo_ai_backlog.csv", mime="text/csv")
        st.markdown("""
<div class='small'>
<b>Implémentations rapides :</b>
<ul>
<li><b>AI1</b> Prévision ventes → job ETL quotidien (`sale.order_line`) + Prophet/XGB → recommandations réassort.</li>
<li><b>AI3</b> Anomalies factures → pipeline `account.move.line` → score d’alerte + workflow 4-yeux.</li>
<li><b>AI6</b> Retards paiement → probabilité & priorisation des relances (<i>account.followup</i>).</li>
<li><b>AI5/AI10</b> RAG + tri tickets → index docs/helpdesk, routage auto par compétence (tags).</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # ---- SLA & Performance
    with q_sla:
        c1,c2,c3 = st.columns(3)
        with c1:
            target = st.slider("SLA cible (%)", 99.0, 99.99, 99.7)
        with c2:
            days = st.slider("Période (jours)", 7, 31, 30)
        with c3:
            budget = error_budget_minutes(target, days)
        st.markdown(f"<div class='badge WARN'>Error budget : {budget} min / {days} j</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_sla_bar(), use_container_width=True, key="sla_bar_d3")
        st.plotly_chart(fig_latency_error(), use_container_width=True, key="lat_err_d3")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("© Odoo — SI — v3.0")

