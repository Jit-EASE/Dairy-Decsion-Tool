#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Irish Dairy Processing Decision Tool — Executive Intelligent Edition
Quantum × AI × Optimisation for Ireland’s dairy sector
Integrates: OR-Tools VRP · QUBO + qLDPC · MILP · (optional) PPO RL · RAG + GPT-4o-mini
TensorBoard is deliberately NOT used to avoid protobuf issues on Python 3.12.
"""

import os, re, math, json
from datetime import datetime
from importlib.util import find_spec
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ========================== App config & header ==========================
st.set_page_config(page_title="Irish Dairy — Executive Intelligent Tool", layout="wide")

st.markdown(
    """
<div style="text-align:center; background:#e8f0fe; padding:1rem; border-radius:12px;">
  <h2>Autonomous Dairy Processing Decision Tool — Ireland </em></h2>
  <p>Self Aware Adaptive Multimodal System (SAAMS) </p>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("Built by Shubhojit Bagchi")

# ---------------- Dependency scan (IMPORT-FREE using find_spec) ----------
def _has_mod(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except Exception:
        return False

HAS_SB3 = _has_mod("stable_baselines3") and _has_mod("gymnasium")
HAS_PULP = _has_mod("pulp")
HAS_SENT = _has_mod("sentence_transformers")
HAS_FAISS = _has_mod("faiss")
HAS_ORT  = _has_mod("ortools")
HAS_PDF  = _has_mod("PyPDF2")
HAS_FOL  = _has_mod("folium") and _has_mod("streamlit_folium")

_missing_labels = []
if not HAS_SB3: _missing_labels.append("RL")
if not HAS_PULP: _missing_labels.append("MILP")
if not (HAS_SENT and HAS_FAISS): _missing_labels.append("RAG(dense)")
if not HAS_ORT: _missing_labels.append("VRP")
if not HAS_PDF: _missing_labels.append("PDF")
if not HAS_FOL: _missing_labels.append("Map")

if _missing_labels:
    st.warning("Optional modules not found: " + ", ".join(sorted(set(_missing_labels))))
if not os.getenv("OPENAI_API_KEY"):
    st.info("OpenAI API key not detected — auto-explanations will use heuristic fallbacks.")

with st.sidebar.expander("Environment check", True):
    st.write({
        "stable_baselines3+gymnasium (RL optional)": HAS_SB3,
        "pulp (MILP)": HAS_PULP,
        "sentence_transformers+faiss (RAG dense)": HAS_SENT and HAS_FAISS,
        "ortools (VRP)": HAS_ORT,
        "PyPDF2 (PDF)": HAS_PDF,
        "folium + streamlit-folium (Map)": HAS_FOL,
        "OPENAI_API_KEY set": bool(os.getenv("OPENAI_API_KEY"))
    })

# =========================== Sidebar navigation ==========================
SECTIONS = [
    "Strategic Network View",
    "Milk Quality & Supply",
    "Market Intelligence & Forecasts",
    "Production Portfolio Optimizer",
    "Learning Allocator",
    "Commercial Plan",
    "Sustainability Dashboard",
    "Scenario Simulator",
    "Intelligent Advisor",
    "Executive Summary & Comparisons",
]
section = st.sidebar.radio("Navigate", SECTIONS, index=0)

st.sidebar.markdown("### Global Controls")
carbon_price = st.sidebar.slider("Carbon price €/t CO₂e", 0.0, 200.0, 60.0, 5.0)
energy_price = st.sidebar.slider("Energy €/kWh", 0.05, 0.30, 0.12, 0.01)
water_price  = st.sidebar.slider("Water €/m³",   0.3,  3.0,  0.8,  0.1)
price_vol    = st.sidebar.slider("Market volatility (Cheddar ln-σ)", 0.05, 0.5, 0.15, 0.01)
service_lvl  = st.sidebar.slider("Forecast Service Level (quantile)", 0.5, 0.95, 0.80, 0.05)
preset       = st.sidebar.selectbox("Preset Strategy", ["Custom", "Max Margin", "Low CO₂e", "Export Push"])

st.sidebar.markdown("### Alerts")
cheddar_trigger = st.sidebar.number_input("Cheddar > €/t triggers replan", 3000, 7000, 5200, 100)
dryer_trigger   = st.sidebar.slider("Dryer utilisation trigger (%)", 50, 100, 90, 5)

# Auto baseline so KPIs update even before running optimizers
auto_from_map = st.sidebar.checkbox(
    "Auto-allocate from map for KPIs",
    True,
    help="Use farms in the current map view to create a baseline mix. Replaced when QUBO/PPO/MILP runs."
)
st.session_state["auto_from_map"] = auto_from_map

# =========================== Synthetic data ==============================
@st.cache_data(show_spinner=False)
def make_synth(seed=42):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")

    states = ["Normal", "Flush", "Drought"]
    P = np.array([[0.8, 0.15, 0.05],
                  [0.2, 0.7, 0.1],
                  [0.25,0.25,0.5]])
    idx = [0]
    for _ in range(len(days) - 1):
        idx.append(int(rng.choice([0,1,2], p=P[idx[-1]])))
    regime = [states[i] for i in idx]

    base_supply = 1.2e6
    mod = {"Normal":1.0, "Flush":1.25, "Drought":0.75}
    fat_base, prot_base = 4.15, 3.45
    fat_mod  = {"Normal":1.00, "Flush":0.97, "Drought":1.03}
    prot_mod = {"Normal":1.00, "Flush":0.99, "Drought":1.02}

    milk_litres = [base_supply*mod[r]*(1+rng.normal(0,0.04)) for r in regime]
    fat_pct     = [fat_base*fat_mod[r]*(1+rng.normal(0,0.01)) for r in regime]
    prot_pct    = [prot_base*prot_mod[r]*(1+rng.normal(0,0.01)) for r in regime]

    milk = pd.DataFrame({
        "date": days,
        "regime": regime,
        "milk_litres": np.array(milk_litres).astype(int),
        "fat_pct": np.round(fat_pct, 3),
        "protein_pct": np.round(prot_pct, 3),
    })

    products = ["Butter", "Cheddar", "WMP", "SMP", "Casein"]
    specs = pd.DataFrame({
        "product": products,
        "base_yield_t_per_kl":[0.42, 0.11, 0.10, 0.08, 0.06],
        "fat_weight":[0.60, 0.05, 0.15, -0.05, 0.00],
        "protein_weight":[0.05, 0.35, 0.05, 0.10, 0.30],
        "energy_kwh_per_t":[650, 900, 1200, 1100, 1500],
        "water_m3_per_t":[5.0, 7.0, 4.0, 4.5, 8.0],
    })

    plant = pd.DataFrame({
        "resource":[
            "proc_capacity_kl","evap_capacity_t","cheese_vat_t",
            "butter_churn_t","dryer_capacity_t","casein_line_t",
            "energy_kwh","water_m3","labour_hours"
        ],
        "value":[1500, 250, 200, 120, 220, 80, 1.5e6, 6000, 3200]
    })

    prices = pd.DataFrame({
        "product": products,
        "price_eur_per_t":[4700, 5200, 3800, 2600, 4200]
    })

    emiss = pd.DataFrame({
        "product": products,
        "co2e_per_t":[1.4, 1.9, 2.6, 2.2, 2.8]
    })

    return milk, specs, plant, prices, emiss

MILK, SPECS, PLANT, PRICES, EMISS = make_synth()
PRODUCTS = SPECS["product"].tolist()

# Apply presets
if preset == "Max Margin":
    carbon_price = 0.0
    energy_price = max(0.10, energy_price)
    water_price  = max(0.50, water_price)
elif preset == "Low CO₂e":
    carbon_price = max(120.0, carbon_price)
elif preset == "Export Push":
    PRICES.loc[PRICES.product.isin(["WMP","SMP"]), "price_eur_per_t"] *= 1.08

# =========================== Session state ==============================
for key, default in [("ctx_alloc", {}), ("ppo_alloc", {}), ("milp_alloc", {}), ("vrp_cost", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================== KPI & helpers ==============================
def yields_t_per_kl_row(row, fat, prot):
    return max(
        row["base_yield_t_per_kl"]
        + row["fat_weight"] * (fat - 4.0) / 4.0
        + row["protein_weight"] * (prot - 3.3) / 3.3,
        1e-6,
    )

def expected_yields_for_window(mdf, specs):
    fat  = float(mdf["fat_pct"].mean())
    prot = float(mdf["protein_pct"].mean())
    y = specs.apply(lambda r: yields_t_per_kl_row(r, fat, prot), axis=1)
    return dict(zip(specs["product"], y))

def kpi_from_allocation(alloc: dict, transport_cost_eur: float = 0.0):
    if not alloc:
        return dict(margin=0, revenue=0, co2e=0, kwh=0, water=0, litres=0)
    price  = dict(zip(PRICES["product"], PRICES["price_eur_per_t"]))
    co2    = dict(zip(EMISS["product"], EMISS["co2e_per_t"]))
    energy = dict(zip(SPECS["product"], SPECS["energy_kwh_per_t"]))
    water  = dict(zip(SPECS["product"], SPECS["water_m3_per_t"]))

    rev = sum(alloc.get(p,0.0) * price[p] for p in PRODUCTS)
    ymap = expected_yields_for_window(MILK, SPECS)
    litres = sum(t / max(ymap.get(p,1e-6), 1e-6) for p, t in alloc.items())
    milk_cost = litres * 0.45
    energy_cost = sum(alloc.get(p,0.0) * energy[p] * energy_price for p in PRODUCTS)
    water_cost  = sum(alloc.get(p,0.0) * water[p]  * water_price  for p in PRODUCTS)

    margin = rev - (milk_cost + energy_cost + water_cost + transport_cost_eur)
    return dict(
        margin=margin, revenue=rev,
        co2e=sum(alloc.get(p,0.0)*co2[p] for p in PRODUCTS),
        kwh=sum(alloc.get(p,0.0)*energy[p] for p in PRODUCTS),
        water=sum(alloc.get(p,0.0)*water[p] for p in PRODUCTS),
        litres=litres
    )

def kpi_strip():
    k = kpi_from_allocation(st.session_state.ctx_alloc, st.session_state.vrp_cost)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Margin (€)", f"{k['margin']:,.0f}")
    c2.metric("Revenue (€)", f"{k['revenue']:,.0f}")
    c3.metric("CO₂e (t)",   f"{k['co2e']:,.1f}")
    c4.metric("Energy (kWh)", f"{k['kwh']:,.0f}")
    c5.metric("Water (m³)",   f"{k['water']:,.0f}")
    return k

def check_alerts(allocation):
    if not allocation:
        return []
    dryer_cap  = float(PLANT.set_index("resource").loc["dryer_capacity_t","value"])
    dryer_util = 100.0 * (allocation.get("WMP",0)+allocation.get("SMP",0)) / max(dryer_cap, 1)
    cheddar_px = float(PRICES.set_index("product").loc["Cheddar","price_eur_per_t"])
    alerts = []
    if cheddar_px > cheddar_trigger:
        alerts.append("Cheddar price above trigger")
    if dryer_util > dryer_trigger:
        alerts.append(f"Dryer utilisation {dryer_util:.0f}%")
    return alerts

# Helper to auto-allocate from synthetic map supply (kl)
def _auto_alloc_from_supply(total_kl: float) -> dict:
    """
    Build a baseline product mix from total milk (kl) using yields and clip by plant capacities.
    This is only used to populate KPIs before any optimizer runs.
    """
    if total_kl <= 0:
        return {}
    ylds = expected_yields_for_window(MILK, SPECS)  # t per kl

    # Neutral product shares (adjust if desired)
    share = {"Butter": 0.25, "Cheddar": 0.25, "WMP": 0.25, "SMP": 0.15, "Casein": 0.10}
    alloc = {p: float(total_kl * ylds[p] * share.get(p, 0.0)) for p in PRODUCTS}

    # Capacity clipping
    caps = PLANT.set_index("resource")["value"].to_dict()
    dryer_cap  = float(caps.get("dryer_capacity_t", 1e12))
    cheese_cap = float(caps.get("cheese_vat_t",     1e12))
    butter_cap = float(caps.get("butter_churn_t",   1e12))
    casein_cap = float(caps.get("casein_line_t",    1e12))

    pair = alloc.get("WMP", 0.0) + alloc.get("SMP", 0.0)
    if pair > dryer_cap and pair > 0:
        scale = dryer_cap / pair
        alloc["WMP"] *= scale
        alloc["SMP"] *= scale

    alloc["Cheddar"] = min(alloc.get("Cheddar", 0.0), cheese_cap)
    alloc["Butter"]  = min(alloc.get("Butter",  0.0), butter_cap)
    alloc["Casein"]  = min(alloc.get("Casein",  0.0), casein_cap)
    return alloc

# --------------------------- County centers + static sites ---------------------------
# Approximate county centroids (deg). Used to generate farms & default placements.
COUNTY_CENTERS = {
    "Carlow": (52.83, -6.93),
    "Cavan": (53.99, -7.36),
    "Clare": (52.86, -8.98),
    "Cork": (51.95, -8.70),
    "Donegal": (54.65, -8.10),
    "Dublin": (53.35, -6.26),
    "Galway": (53.27, -9.06),
    "Kerry": (52.27, -9.70),
    "Kildare": (53.16, -6.91),
    "Kilkenny": (52.65, -7.25),
    "Laois": (53.03, -7.30),
    "Leitrim": (54.13, -8.00),
    "Limerick": (52.66, -8.63),
    "Longford": (53.73, -7.80),
    "Louth": (53.95, -6.54),
    "Mayo": (53.85, -9.30),
    "Meath": (53.65, -6.65),
    "Monaghan": (54.25, -6.97),
    "Offaly": (53.27, -7.49),
    "Roscommon": (53.63, -8.20),
    "Sligo": (54.27, -8.47),
    "Tipperary": (52.60, -8.00),
    "Waterford": (52.26, -7.11),
    "Westmeath": (53.54, -7.35),
    "Wexford": (52.34, -6.46),
    "Wicklow": (52.99, -6.35),
}

# Major ports around Ireland (approximate coordinates).
PORTS_STATIC = [
    {"name":"Dublin Port","lat":53.345,"lon":-6.215},
    {"name":"Cork (Ringaskiddy)","lat":51.810,"lon":-8.300},
    {"name":"Shannon Foynes","lat":52.610,"lon":-9.110},
    {"name":"Waterford (Belview)","lat":52.250,"lon":-7.000},
    {"name":"Rosslare Europort","lat":52.260,"lon":-6.340},
    {"name":"Galway Port","lat":53.270,"lon":-9.050},
    {"name":"Killybegs","lat":54.630,"lon":-8.450},
    {"name":"Drogheda","lat":53.720,"lon":-6.250},
    {"name":"Dundalk","lat":54.000,"lon":-6.380},
    {"name":"New Ross","lat":52.400,"lon":-6.950},
    {"name":"Bantry (Whiddy)","lat":51.680,"lon":-9.450},
    {"name":"Sligo Port","lat":54.270,"lon":-8.470},
]

# Example processor sites (representative, approximate).
PROCESSORS_STATIC = [
    {"name":"Mallow Processor","county":"Cork","lat":52.137,"lon":-8.636},
    {"name":"Mitchelstown Processor","county":"Cork","lat":52.270,"lon":-8.270},
    {"name":"Ballineen Processor","county":"Cork","lat":51.720,"lon":-9.110},
    {"name":"Listowel Processor","county":"Kerry","lat":52.440,"lon":-9.490},
    {"name":"Nenagh Processor","county":"Tipperary","lat":52.860,"lon":-8.200},
    {"name":"Ballyragget Processor","county":"Kilkenny","lat":52.780,"lon":-7.350},
    {"name":"Belview Processor","county":"Kilkenny","lat":52.250,"lon":-7.000},
    {"name":"Charleville Processor","county":"Limerick","lat":52.350,"lon":-8.670},
    {"name":"Ballaghaderreen Processor","county":"Roscommon","lat":53.900,"lon":-8.580},
    {"name":"Killeshandra Processor","county":"Cavan","lat":54.050,"lon":-7.430},
    {"name":"Virginia Processor","county":"Cavan","lat":53.830,"lon":-7.080},
]

# =========================== QUBO + qLDPC ===============================
def build_parity_matrix(n_bits, check_density=0.25, seed=123):
    rng = np.random.default_rng(seed)
    H = np.zeros((max(3, int(n_bits*0.3)), n_bits), dtype=int)
    for i in range(H.shape[0]):
        H[i, rng.choice(n_bits, size=max(1,int(check_density*n_bits)), replace=False)] = 1
    return H

def qubo_from_all(products, price_map, caps, K=6, Tmax=300.0,
                  ldpc_weight=5.0, cap_weight=1.0,
                  carbon_price=0.0, energy_price=0.0, water_price=0.0):
    P = len(products); n_bits = P*K; step = Tmax/(2**K - 1)
    bit_index = {}; idx = 0
    for p in products:
        for k in range(K):
            bit_index[idx] = (p, k); idx += 1

    Q = {}
    energy = dict(zip(SPECS["product"], SPECS["energy_kwh_per_t"]))
    water  = dict(zip(SPECS["product"], SPECS["water_m3_per_t"]))
    emiss  = dict(zip(EMISS["product"], EMISS["co2e_per_t"]))

    # Objective: profit (price minus resource "shadow" prices)
    for i in range(n_bits):
        p, k = bit_index[i]
        v = price_map.get(p,0.0) - (carbon_price*emiss[p] + energy_price*energy[p] + water_price*water[p])
        Q[(i,i)] = Q.get((i,i), 0.0) - v * step * (2**k)

    # Capacity penalties
    def add_capacity(subset, cap, w):
        a = np.zeros(n_bits)
        for i in range(n_bits):
            p, k = bit_index[i]
            if p in subset:
                a[i] = step * (2**k)
        for i in range(n_bits):
            for j in range(i, n_bits):
                Q[(i,j)] = Q.get((i,j), 0.0) + w*(a[i]*a[j])
        for i in range(n_bits):
            Q[(i,i)] = Q.get((i,i), 0.0) - 2*w*cap*a[i]

    add_capacity(["WMP","SMP"], caps.get("dryer_capacity_t",0.0), cap_weight)
    add_capacity(["Cheddar"],   caps.get("cheese_vat_t",0.0),   cap_weight)
    add_capacity(["Butter"],    caps.get("butter_churn_t",0.0), cap_weight)
    add_capacity(["Casein"],    caps.get("casein_line_t",0.0),  cap_weight)

    # LDPC stabilisers (gentle)
    H = build_parity_matrix(n_bits, check_density=0.25, seed=123)
    for r in range(H.shape[0]):
        row = np.where(H[r]==1)[0]
        for i in range(len(row)):
            for j in range(i, len(row)):
                Q[(int(row[i]), int(row[j]))] = Q.get((int(row[i]), int(row[j])), 0.0) + ldpc_weight*0.01

    def decode(bits):
        alloc = {p:0.0 for p in products}
        for i, b in enumerate(bits):
            if b > 0.5:
                p, k = bit_index[i]
                alloc[p] += step * (2**k)
        return alloc

    return Q, decode, H

def solve_qubo(Q, H=None, max_iter=3500, beta=0.995):
    # Try dimod if present
    try:
        import dimod
        from dimod import SimulatedAnnealingSampler
        bqm  = dimod.BinaryQuadraticModel.from_qubo(Q)
        resp = SimulatedAnnealingSampler().sample(bqm, num_reads=50)
        s    = resp.first.sample; E = resp.first.energy
        bits = np.array([s[i] for i in range(len(s))])
        return bits, E, "dimod-SA"
    except Exception:
        pass

    # Fallback: simple Simulated Annealing
    rng = np.random.default_rng(123)
    n   = max([max(i,j) for (i,j) in Q.keys()]) + 1
    x   = rng.integers(0, 2, size=n)

    def energy(xv):
        E = 0.0
        for (i,j), c in Q.items():
            E += c * xv[i] * xv[j]
        if H is not None:
            E += 0.1 * np.sum((H @ xv) % 2)
        return E

    E = energy(x); T = 1.0
    for _ in range(max_iter):
        i = rng.integers(0, n)
        xn = x.copy(); xn[i] = 1 - xn[i]
        En = energy(xn)
        if En < E or rng.random() < np.exp((E - En)/T):
            x, E = xn, En
        T *= beta
    return x, E, "internal-SA"

# =========================== RL (PPO) — optional ========================
def ppo_train_and_save(model_path="ppo_alloc.zip", timesteps=15000):
    """
    Trains PPO if available. Never touches TensorBoard. If RL stack
    is unavailable, returns a clear error dict (UI handles it).
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        import gymnasium as gym
    except Exception as e:
        return {"status":"error", "error": f"RL unavailable: {e}"}

    price_map = dict(zip(PRICES["product"], PRICES["price_eur_per_t"]))
    caps      = {r["resource"]: float(r["value"]) for _, r in PLANT.iterrows()}
    ylds      = expected_yields_for_window(MILK, SPECS)

    class Env(gym.Env):
        metadata = {"render_modes":[]}
        def __init__(self):
            super().__init__()
            self.products = PRODUCTS
            self.action_space = gym.spaces.Box(0.0, 1.0, shape=(len(PRODUCTS),), dtype=np.float32)
            low  = np.zeros(len(PRODUCTS)+6, dtype=np.float32)
            high = np.ones (len(PRODUCTS)+6, dtype=np.float32) * 1e6
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        def reset(self, seed=None, options=None):
            vol = 0.15
            self.price = {p: price_map[p] * np.exp(np.random.normal(-0.5*vol**2, vol))
                          for p in self.products}
            fat  = 4.0 + np.random.normal(0, 0.1)
            prot = 3.3 + np.random.normal(0, 0.05)
            obs = np.array(
                [fat, prot] + [self.price[p] for p in self.products] +
                [caps["dryer_capacity_t"], caps["cheese_vat_t"], caps["butter_churn_t"], caps["casein_line_t"]],
                dtype=np.float32
            )
            return obs, {}

        def step(self, action):
            w = np.maximum(action, 0.0); w = w / (w.sum() + 1e-8)
            total_t = sum([ylds[p] * 1000 for p in self.products])
            alloc = {p: float(w[i] * total_t) for i, p in enumerate(self.products)}

            over = lambda x, c: max(0.0, x - c)
            pen = (
                over(alloc["WMP"]+alloc["SMP"], caps["dryer_capacity_t"])**2
                + over(alloc["Cheddar"], caps["cheese_vat_t"])**2
                + over(alloc["Butter"],  caps["butter_churn_t"])**2
                + over(alloc["Casein"],  caps["casein_line_t"])**2
            )
            rev = sum([alloc[p] * self.price[p] for p in self.products])
            reward = rev - 1000.0 * pen
            return self.reset()[0], reward, True, False, {"alloc": alloc}

    vec   = make_vec_env(Env, n_envs=4, seed=123)
    model = PPO("MlpPolicy", vec, verbose=0)  # no tensorboard_log
    model.learn(total_timesteps=int(timesteps))
    model.save(model_path)
    return {"status":"ok", "timesteps": int(timesteps)}

def ppo_infer_allocation(model_path="ppo_alloc.zip"):
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        return {"_error": f"RL unavailable: {e}"}
    if not os.path.exists(model_path):
        return {"_error": "Model file not found. Train first."}

    model = PPO.load(model_path)
    price_map = dict(zip(PRICES["product"], PRICES["price_eur_per_t"]))
    caps      = {r["resource"]: float(r["value"]) for _, r in PLANT.iterrows()}
    ylds      = expected_yields_for_window(MILK, SPECS)

    obs = np.array(
        [MILK["fat_pct"].mean(), MILK["protein_pct"].mean()]
        + [price_map[p] for p in PRODUCTS]
        + [caps["dryer_capacity_t"], caps["cheese_vat_t"], caps["butter_churn_t"], caps["casein_line_t"]],
        dtype=np.float32
    ).reshape(1, -1)

    action, _ = model.predict(obs, deterministic=True)
    w = np.maximum(action.flatten(), 0.0); w = w/(w.sum()+1e-8)
    total_t = sum([ylds[p] * 1000 for p in PRODUCTS])
    return {p: float(w[i]*total_t) for i, p in enumerate(PRODUCTS)}

# =========================== MILP (PuLP) ================================
def milp_solve(dates, demand_df, cap_map, price_map):
    try:
        import pulp
    except Exception as e:
        return {"status":"error", "error": f"PuLP not available: {e}"}

    prods = demand_df["product"].unique().tolist()
    T = list(range(len(dates)))
    prob = pulp.LpProblem("Planner", pulp.LpMaximize)

    prod = pulp.LpVariable.dicts("prod", (prods, T), lowBound=0)
    inv  = pulp.LpVariable.dicts("inv",  (prods, T), lowBound=0)
    ship = pulp.LpVariable.dicts("ship", (prods, T), lowBound=0)

    energy = dict(zip(SPECS["product"], SPECS["energy_kwh_per_t"]))
    water  = dict(zip(SPECS["product"], SPECS["water_m3_per_t"]))
    co2    = dict(zip(EMISS["product"], EMISS["co2e_per_t"]))

    def unit_margin(p):
        return price_map.get(p,0.0) - (
            energy_price*energy[p] + water_price*water[p] + carbon_price*co2[p]
        )

    prob += pulp.lpSum([ship[p][t] * unit_margin(p) for p in prods for t in T])

    dryer = cap_map.get("dryer_capacity_t", 220.0)
    cheese= cap_map.get("cheese_vat_t",     200.0)
    butter= cap_map.get("butter_churn_t",   120.0)
    casein= cap_map.get("casein_line_t",     80.0)
    for t in T:
        prob += prod["WMP"][t] + prod["SMP"][t] <= dryer
        prob += prod["Cheddar"][t] <= cheese
        prob += prod["Butter"][t]  <= butter
        prob += prod["Casein"][t]  <= casein

    dem = demand_df.set_index(["product","date"])["demand_t"].to_dict()
    for p in prods:
        for t in T:
            d = float(dem.get((p, dates[t]), 0.0))
            if t == 0:
                prob += prod[p][t] - ship[p][t] - inv[p][t] == 0
            else:
                prob += prod[p][t] + inv[p][t-1] - ship[p][t] - inv[p][t] == 0
            prob += ship[p][t] <= d

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = {"status": pulp.LpStatus[prob.status], "objective": pulp.value(prob.objective)}
    if sol["status"] != "Optimal":
        return sol
    sol["dates"] = [str(d) for d in dates]
    sol["ship"]  = {p: [ship[p][t].value() or 0 for t in T] for p in prods}
    sol["inv"]   = {p: [inv[p][t].value()  or 0 for t in T] for p in prods}
    return sol

# =========================== Forecasting ================================
def forecast_series(y: pd.Series, horizon=8, alpha=0.25):
    if y.empty:
        return np.zeros(horizon), np.zeros(horizon), np.zeros(horizon)
    level = y.iloc[0]
    for v in y.values:
        level = alpha*v + (1-alpha)*level
    mean = np.ones(horizon) * level
    sigma = 0.15
    p10 = mean * np.exp(-1.2816*sigma)
    p90 = mean * np.exp( 1.2816*sigma)
    return mean, p10, p90

# =========================== VRP (OR-Tools) =============================
def haversine(lat1, lon1, lat2, lon2):
    R=6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl   = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def solve_vrp_ortools(coops, plants, ports, vehicle_capacity_kl=25000, km_cost=1.6):
    if not HAS_ORT:
        return None
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None
    if coops.empty or plants.empty:
        return None

    depot_row = plants.iloc[0]
    nodes = pd.concat([pd.DataFrame([depot_row]), coops], ignore_index=True)
    N = len(nodes)

    dist_km = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            dist_km[i,j] = 0 if i==j else haversine(nodes.lat[i], nodes.lon[i], nodes.lat[j], nodes.lon[j])

    demands = [0.0] + list(coops["demand_kl"].astype(float))
    manager = pywrapcp.RoutingIndexManager(N, 5, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(fi, ti):
        i, j = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return int(dist_km[i,j] * 1000)
    transit_cb = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    demand_cb = routing.RegisterUnaryTransitCallback(lambda idx: int(demands[manager.IndexToNode(idx)]))
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, [int(vehicle_capacity_kl)]*5, True, "Capacity")

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(5)

    sol = routing.SolveWithParameters(p)
    if not sol:
        return None

    legs, total_km = [], 0.0
    for v in range(5):
        index = routing.Start(v)
        while not routing.IsEnd(index):
            n1 = manager.IndexToNode(index)
            n2 = manager.IndexToNode(sol.Value(routing.NextVar(index)))
            if n1 != n2:
                dkm = dist_km[n1, n2]
                legs.append({"from": int(n1), "to": int(n2), "km": float(dkm)})
                total_km += dkm
            index = sol.Value(routing.NextVar(index))
    return {"legs": pd.DataFrame(legs), "total_km": total_km, "transport_cost_eur": total_km * km_cost}

def render_geospatial_header(county_filter: str = "All"):
    """
    Draws an interactive map with farms → co-ops → processors → ports.
    - County dropdown filters farms & co-ops. Processors/ports remain visible.
    - Connections are nearest-neighbour paths:
        Farm -> nearest Co-op (prefer same county)
        Co-op -> nearest Processor
        Processor -> nearest Port
    Also updates st.session_state.vrp_cost using a simple €/km cost.
    """
    if not HAS_FOL:
        st.info("Install folium + streamlit-folium for the interactive map.")
        st.session_state.vrp_cost = 0.0
        return

    import folium
    from streamlit_folium import st_folium

    rng = np.random.default_rng(7)

    # Static layers
    ports = pd.DataFrame(PORTS_STATIC)
    processors = pd.DataFrame(PROCESSORS_STATIC)

    # Generate synthetic co-ops: 2–3 per county around centroid
    coop_rows = []
    for cnt, (clat, clon) in COUNTY_CENTERS.items():
        for i, (dlat, dlon) in enumerate([(0.10, 0.10), (-0.08, 0.12), (0.06, -0.10)], start=1):
            coop_rows.append({
                "name": f"{cnt} Co-op {i}",
                "county": cnt,
                "lat": clat + dlat,
                "lon": clon + dlon,
            })
    coops = pd.DataFrame(coop_rows)

    # Generate synthetic farms: up to 25 per county, random jitter near centroid
    farm_rows = []
    for cnt, (clat, clon) in COUNTY_CENTERS.items():
        n = 25 if county_filter in ("All", cnt) else 8  # keep map light outside selected county
        for i in range(n):
            farm_rows.append({
                "name": f"{cnt} Farm {i+1}",
                "county": cnt,
                "lat": float(clat + rng.uniform(-0.22, 0.22)),
                "lon": float(clon + rng.uniform(-0.22, 0.22)),
            })
    farms = pd.DataFrame(farm_rows)

    # Filter by county where applicable (farms & co-ops)
    if county_filter and county_filter != "All":
        farms = farms[farms["county"] == county_filter].reset_index(drop=True)
        coops = coops[coops["county"] == county_filter].reset_index(drop=True)

    # Base map
    m = folium.Map(location=[53.4, -8.0], zoom_start=6, tiles="OpenStreetMap")

    # Feature groups
    fg_farms = folium.FeatureGroup(name="Farms", show=True)
    fg_coops = folium.FeatureGroup(name="Co-ops", show=True)
    fg_procs = folium.FeatureGroup(name="Processors", show=True)
    fg_ports = folium.FeatureGroup(name="Ports", show=True)
    fg_edges = folium.FeatureGroup(name="Connections", show=True)

    # Add markers
    for _, r in ports.iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=6, color="blue", fill=True, fill_opacity=0.9,
            popup=f"Port: {r.name}"
        ).add_to(fg_ports)

    for _, r in processors.iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=7, color="red", fill=True, fill_opacity=0.9,
            popup=f"Processor: {r.name} ({r.county})"
        ).add_to(fg_procs)

    for _, r in coops.iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=6, color="orange", fill=True, fill_opacity=0.9,
            popup=f"Co-op: {r.name} ({r.county})"
        ).add_to(fg_coops)

    for _, r in farms.iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=4, color="green", fill=True, fill_opacity=0.8,
            popup=f"Farm: {r.name} ({r.county})"
        ).add_to(fg_farms)

    # Nearest-neighbour helpers
    def _nearest(lat, lon, df):
        if df.empty:
            return None, None, None
        dists = df.apply(lambda row: haversine(lat, lon, row.lat, row.lon), axis=1)
        idx = dists.idxmin()
        dist = float(dists.loc[idx])
        row = df.loc[idx]
        return (row.lat, row.lon, dist)

    total_km = 0.0
    used_processor_idx = set()

    # Connect farms -> co-ops
    for _, f in farms.iterrows():
        # Prefer co-ops in same county; fallback to any co-op if filtered set empty
        same_cnt = coops[coops["county"] == f.county] if not coops.empty else pd.DataFrame()
        candidates = same_cnt if not same_cnt.empty else coops
        if candidates.empty:
            continue
        lat2, lon2, dkm = _nearest(f.lat, f.lon, candidates)
        folium.PolyLine([[f.lat, f.lon], [lat2, lon2]], color="green", weight=2, opacity=0.6).add_to(fg_edges)
        total_km += dkm

    # Connect co-ops -> processors
    for idx, c in coops.iterrows():
        lat2, lon2, dkm = _nearest(c.lat, c.lon, processors)
        if lat2 is None:
            continue
        folium.PolyLine([[c.lat, c.lon], [lat2, lon2]], color="orange", weight=2.5, opacity=0.7).add_to(fg_edges)
        total_km += dkm
        # Record nearest processor (match by coordinates)
        near = processors.apply(lambda r: math.isclose(r.lat, lat2, abs_tol=1e-6) and math.isclose(r.lon, lon2, abs_tol=1e-6), axis=1)
        if near.any():
            used_processor_idx.add(int(near[near].index[0]))

    # Connect processors -> ports (only for processors used by any co-op)
    proc_subset = processors.iloc[list(used_processor_idx)] if used_processor_idx else processors
    for _, p in proc_subset.iterrows():
        lat2, lon2, dkm = _nearest(p.lat, p.lon, ports)
        if lat2 is None:
            continue
        folium.PolyLine([[p.lat, p.lon], [lat2, lon2]], color="red", weight=3, opacity=0.65).add_to(fg_edges)
        total_km += dkm

    # Add groups + control
    fg_farms.add_to(m); fg_coops.add_to(m); fg_procs.add_to(m); fg_ports.add_to(m); fg_edges.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Update a simple transport cost proxy (€/km)
    km_cost = 1.6
    st.session_state.vrp_cost = float(total_km * km_cost)

    # Expose a synthetic milk supply (kl) based on farms shown vs. "all farms"
    try:
        avg_daily_kl = float(MILK["milk_litres"].mean() / 1000.0)  # average day, kl
        farms_all = 25 * len(COUNTY_CENTERS)  # we generate 25 farms per county in total view
        scaler = len(farms) / float(farms_all) if farms_all > 0 else 0.0
        total_kl = max(0.0, avg_daily_kl * scaler)
    except Exception:
        # Fallback: assign 30 kl per farm
        total_kl = float(len(farms)) * 30.0
    st.session_state["_map_supply_kl"] = total_kl

    st_folium(m, width=None, height=460)

# =========================== RAG + GPT ==================================
def rag_embed_texts(texts):
    if HAS_SENT and HAS_FAISS:
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
            index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs)
            return ("dense", model, index, embs)
        except Exception:
            pass
    # TF-IDF fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=8000)
    tfidf = vec.fit_transform(texts)
    return ("tfidf", vec, tfidf, None)

def rag_search(query, index_tuple, texts, names, k=4):
    kind, A, B, E = index_tuple
    if kind == "dense":
        model, index = A, B
        q = model.encode([query or ""], normalize_embeddings=True).astype("float32")
        D, I = index.search(q, min(k, len(texts)))
        return [(names[i], texts[i], float(D[0][j])) for j,i in enumerate(I[0]) if i != -1]
    else:
        from sklearn.metrics.pairwise import linear_kernel
        vec, tfidf = A, B
        q  = vec.transform([query or ""])
        sc = linear_kernel(q, tfidf).flatten()
        order = np.argsort(sc)[-k:][::-1]
        return [(names[i], texts[i], float(sc[i])) for i in order if sc[i] > 0]

def gpt_decide(query, context):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        msgs = [
            {"role":"system","content":"You are an optimisation strategist. Reply with 'PREFERS=QUBO|PPO|MILP' and a short reason."},
            {"role":"user","content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
        r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.3, max_tokens=200)
        text = r.choices[0].message.content.strip()
    except Exception:
        dryer = float(PLANT.set_index("resource").loc["dryer_capacity_t","value"])
        pref  = "QUBO" if dryer < 180 else "MILP"
        text  = f"PREFERS={pref} (heuristic fallback)"
    m = re.search(r"PREFERS\s*=\s*(QUBO|PPO|MILP)", text, re.I)
    return {"text": text, "trigger": (m.group(1).upper() if m else None)}

def explain_section(title, context_text, data_summary=""):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""You are an economics strategist. ≤100 words, board-style insight.
Title: {title}
Context: {context_text}
Data Summary: {data_summary}
No code or formulas; state implications, risks, opportunities."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a strategic business analyst."},
                      {"role":"user","content":prompt}],
            temperature=0.35, max_tokens=180
        )
        st.info(resp.choices[0].message.content.strip())
    except Exception:
        st.info(f"({title}) Insight: Configure OPENAI_API_KEY for auto-explanations.")

# =========================== Geospatial header + KPIs ====================
# County filter for the map
county_filter = st.sidebar.selectbox("County filter", ["All"] + sorted(COUNTY_CENTERS.keys()))
render_geospatial_header(county_filter)

# Auto baseline allocation for KPIs from the map view
if st.session_state.get("auto_from_map", False):
    total_kl = float(st.session_state.get("_map_supply_kl", 0.0))
    if total_kl > 0 and (not st.session_state.ctx_alloc or st.session_state.get("_alloc_source") == "auto"):
        st.session_state.ctx_alloc = _auto_alloc_from_supply(total_kl)
        st.session_state["_alloc_source"] = "auto"

k_now = kpi_strip()
alerts = check_alerts(st.session_state.ctx_alloc)
if alerts:
    st.warning(" | ".join(alerts))
st.divider()

# =========================== Tabs / Sections =============================
# --- Strategic Network View ---
if section == "Strategic Network View":
    st.header("Strategic Network View")
    st.markdown("Optimise routes from co-ops → processors → ports using OR-Tools VRP.")
    st.metric("VRP Transport Cost (€)", f"{st.session_state.vrp_cost:,.0f}")
    explain_section(section, "Geospatial network + routing cost exposure from VRP.")

# --- Milk Quality & Supply ---
elif section == "Milk Quality & Supply":
    st.header("Milk Quality & Supply")
    fig = px.area(MILK, x="date", y="milk_litres", color="regime",
                  title="Daily Milk Supply (2024)", labels={"milk_litres":"Litres"})
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Mean Fat %",     f"{MILK['fat_pct'].mean():.2f}")
    c2.metric("Mean Protein %", f"{MILK['protein_pct'].mean():.2f}")
    explain_section(section, "Seasonal variation in raw milk volume and composition informs yield & mix.")

# --- Market Intelligence & Forecasts ---
elif section == "Market Intelligence & Forecasts":
    st.header("Market Intelligence & Forecasts")
    base  = float(PRICES.set_index("product").loc["Cheddar","price_eur_per_t"])
    draws = base * np.exp(np.random.normal(-0.5*price_vol**2, price_vol, size=2000))
    h = np.histogram(draws, bins=50)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=0.5*(h[1][1:]+h[1][:-1]), y=h[0]))
    fig.update_layout(title="Cheddar Price Distribution (lognormal)", xaxis_title="€/t", yaxis_title="Freq")
    st.plotly_chart(fig, use_container_width=True)

    weekly_kl = (MILK.resample("W", on="date")["milk_litres"].sum() / 1e3)
    mean, p10, p90 = forecast_series(weekly_kl, horizon=8)
    fdf = pd.DataFrame({
        "week": pd.date_range(weekly_kl.index[-1] + pd.Timedelta(weeks=1), periods=8, freq="W"),
        "mean": mean, "p10": p10, "p90": p90
    })
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fdf["week"], y=fdf["mean"], name="Forecast"))
    fig2.add_trace(go.Scatter(x=fdf["week"], y=fdf["p10"],  name="p10", line=dict(dash="dot")))
    fig2.add_trace(go.Scatter(x=fdf["week"], y=fdf["p90"],  name="p90", line=dict(dash="dot")))
    fig2.update_layout(title="Weekly Intake Forecast (kl)", xaxis_title="", yaxis_title="kl")
    st.plotly_chart(fig2, use_container_width=True)
    explain_section(section, "Commodity volatility + intake forecasts set near-term capacity & mix expectations.")

# --- Production Portfolio Optimizer (QUBO) ---
elif section == "Production Portfolio Optimizer":
    st.header("Production Portfolio Optimizer (QUBO + qLDPC)")
    price_map = dict(zip(PRICES["product"], PRICES["price_eur_per_t"]))
    caps      = {r["resource"]: float(r["value"]) for _, r in PLANT.iterrows()}
    with st.spinner("Solving QUBO..."):
        Q, decode, H = qubo_from_all(PRODUCTS, price_map, caps, K=6, Tmax=300, ldpc_weight=5.0,
                                     carbon_price=carbon_price, energy_price=energy_price, water_price=water_price)
        bits, E, solver = solve_qubo(Q, H)
        alloc = decode(bits)
    st.session_state.ctx_alloc = alloc
    st.session_state["_alloc_source"] = "qubo"
    df = pd.DataFrame(list(alloc.items()), columns=["Product","Tonnes"]).sort_values("Tonnes", ascending=False)
    st.plotly_chart(px.bar(df, x="Product", y="Tonnes", title=f"Recommended Mix (QUBO via {solver})"),
                    use_container_width=True)
    st.json(kpi_from_allocation(alloc, st.session_state.vrp_cost))
    explain_section(section, "Quantum-inspired mix optimiser balancing profitability & sustainability costs.")

# --- Learning Allocator (AI, optional) ---
elif section == "Learning Allocator (AI, optional)":
    st.header("Learning Allocator (PPO Reinforcement Learning)")
    if not HAS_SB3:
        st.error(
            "Reinforcement Learning is unavailable in this environment.\n\n"
            "Quick fix (Python 3.12):\n"
            "1) Uninstall TB/TF: `pip uninstall -y tensorboard tb-nightly tensorboard-data-server tensorflow tensorflow-cpu`\n"
            "2) Install RL stack: `pip install -U stable-baselines3==2.3.2 gymnasium==0.29.1 torch --index-url https://download.pytorch.org/whl/cpu`"
        )
    else:
        steps = st.number_input("Training steps", min_value=2000, step=1000, value=15000)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train PPO (no TensorBoard)"):
                with st.spinner("Training PPO..."):
                    st.write(ppo_train_and_save("ppo_alloc.zip", timesteps=int(steps)))
        with col2:
            if st.button("Apply Learned Policy"):
                alloc = ppo_infer_allocation("ppo_alloc.zip")
                if "_error" in alloc:
                    st.error(alloc["_error"])
                else:
                    st.session_state.ctx_alloc = alloc
                    st.session_state.ppo_alloc = alloc
                    st.session_state["_alloc_source"] = "ppo"
                    df = pd.DataFrame(list(alloc.items()), columns=["Product","Tonnes"])
                    st.plotly_chart(px.bar(df, x="Product", y="Tonnes", title="Learning Policy Mix"),
                                    use_container_width=True)
                    st.json(kpi_from_allocation(alloc, st.session_state.vrp_cost))
    explain_section(section, "RL adapts to stochastic prices/capacities; logging avoids TensorBoard entirely.")

# --- Commercial Plan (MILP) ---
elif section == "Commercial Plan (MILP)":
    st.header("Commercial Production Planner (MILP)")
    weekly_kl = (MILK.resample("W", on="date")["milk_litres"].sum()/1e3)
    mean, p10, p90 = forecast_series(weekly_kl, horizon=6)
    q = np.clip(service_lvl, 0.1, 0.9)
    dem_kl = mean + (q-0.5)/(0.9-0.1) * (p90 - p10)
    horizon = pd.date_range(weekly_kl.index[-1] + pd.Timedelta(weeks=1), periods=6, freq="W")

    ylds = expected_yields_for_window(MILK, SPECS)
    rows = []
    for t_idx, dkl in enumerate(dem_kl):
        for p in PRODUCTS:
            rows.append({"date": horizon[t_idx], "product": p, "demand_t": float(max(0.0, dkl*ylds[p]))})
    demand = pd.DataFrame(rows)

    caps      = {r["resource"]: float(r["value"]) for _, r in PLANT.iterrows()}
    price_map = dict(zip(PRICES["product"], PRICES["price_eur_per_t"]))
    with st.spinner("Solving MILP..."):
        sol = milp_solve(horizon, demand, caps, price_map)

    if sol.get("status") == "Optimal":
        df_ship = pd.DataFrame(sol["ship"])
        st.plotly_chart(px.line(df_ship, title="Shipments (t)"), use_container_width=True)
        st.success(f"Objective = €{sol['objective']:.0f}")
        tot = pd.DataFrame(sol["ship"]).sum(axis=0).to_dict()
        st.session_state.ctx_alloc = {k: float(v) for k, v in tot.items()}
        st.session_state.milp_alloc = st.session_state.ctx_alloc
        st.session_state["_alloc_source"] = "milp"
    else:
        st.error(f"Solve status: {sol.get('status')}")
    explain_section(section, "MILP balances production/inventory/demand with sustainability-adjusted margins.")

# --- Sustainability Dashboard ---
elif section == "Sustainability Dashboard":
    st.header("Sustainability Frontier (Multi-price Sweep)")
    caps = {r["resource"]: float(r["value"]) for _, r in PLANT.iterrows()}
    base_pm = dict(zip(PRICES["product"], PRICES["price_eur_per_t"]))
    grid = []
    with st.spinner("Sweeping carbon/energy/water prices..."):
        for cp in np.linspace(0, 200, 6):
            for ep in np.linspace(max(0.08, energy_price), max(0.24, energy_price), 4):
                for wp in np.linspace(max(0.5, water_price), max(2.0, water_price), 4):
                    Q, decode, H = qubo_from_all(
                        PRODUCTS, base_pm, caps, K=6, Tmax=300, ldpc_weight=5.0,
                        carbon_price=float(cp), energy_price=float(ep), water_price=float(wp)
                    )
                    bits, _, _ = solve_qubo(Q, H, max_iter=1600)
                    alloc = decode(bits); k = kpi_from_allocation(alloc)
                    grid.append({
                        "Revenue":k["revenue"], "CO₂e":k["co2e"], "Energy":k["kwh"], "Water":k["water"],
                        "CO2€":cp, "kWh€":ep, "H2O€":wp
                    })
    df = pd.DataFrame(grid)
    fig = px.scatter_3d(df, x="CO₂e", y="Energy", z="Revenue", color="CO2€",
                        hover_data=["kWh€","H2O€","Water"], title="Revenue × CO₂e × Energy Frontier")
    st.plotly_chart(fig, use_container_width=True)
    explain_section(section, "Frontier reveals profitable yet resource-efficient operating points across price regimes.")

# --- Scenario Simulator ---
elif section == "Scenario Simulator":
    st.header("Scenario Simulator (Monte Carlo)")
    n = 400
    shocks = np.random.normal(0, price_vol, n)
    df = pd.DataFrame({"Shock":shocks})
    st.plotly_chart(px.histogram(df, x="Shock", nbins=30, title="Price Shock Distribution"),
                    use_container_width=True)
    explain_section(section, "Stress-tests outcomes against price volatility and demand/capacity shocks.")

# --- Intelligent Advisor (RAG + GPT) ---
elif section == "Intelligent Advisor (RAG + GPT)":
    st.header("Intelligent Advisor (Auto RAG + GPT-4o-mini)")
    uploaded = st.file_uploader("Upload industry reports (PDF / TXT)", type=["pdf","txt"], accept_multiple_files=True)
    texts, names = [], []

    # built-in data tables as seeds
    base_sources = [("milk.csv", MILK.head(80).to_csv(index=False)),
                    ("specs.csv", SPECS.to_csv(index=False)),
                    ("plant.csv", PLANT.to_csv(index=False)),
                    ("prices.csv", PRICES.to_csv(index=False)),
                    ("emissions.csv", EMISS.to_csv(index=False))]
    bn, bt = zip(*base_sources); names += list(bn); texts += list(bt)

    if uploaded:
        for f in uploaded:
            try:
                if HAS_PDF:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(f)
                    t = " ".join(page.extract_text() or "" for page in reader.pages)
                else:
                    raise RuntimeError("PyPDF2 not available")
            except Exception:
                t = f.read().decode(errors="ignore")
            texts.append(t[:8000]); names.append(f.name)

    if texts:
        idx = rag_embed_texts(texts)
        q = st.text_input("Ask a strategic question:", value="Which approach fits our current constraints and prices?")
        if q:
            res = rag_search(q, idx, texts, names, k=4)
            st.markdown("**Top sources:** " + " · ".join([f"`{r[0]}`" for r in res]))
            ctx = "\n\n".join([f"[{i+1}] {res[i][1][:400]}" for i in range(len(res))])
            dec = gpt_decide(q, ctx)
            st.success(dec["text"])
    else:
        st.info("Upload documents or rely on built-in data tables (already loaded).")
    explain_section(section, "Retrieves evidence from data & docs, then selects QUBO/PPO/MILP with rationale.")

# --- Executive Summary & Comparisons ---
elif section == "Executive Summary & Comparisons":
    st.header("Executive Summary & Comparisons")
    comp = []
    if st.session_state.ctx_alloc:
        for p,t in st.session_state.ctx_alloc.items():
            comp.append({"Method":"Current","Product":p,"Tonnes":float(t)})
    if st.session_state.ppo_alloc:
        for p,t in st.session_state.ppo_alloc.items():
            comp.append({"Method":"PPO","Product":p,"Tonnes":float(t)})
    if st.session_state.milp_alloc:
        for p,t in st.session_state.milp_alloc.items():
            comp.append({"Method":"MILP","Product":p,"Tonnes":float(t)})

    if comp:
        dfc = pd.DataFrame(comp)
        st.plotly_chart(
            px.bar(dfc, x="Product", y="Tonnes", color="Method", barmode="group",
                   title="Allocations by Method"),
            use_container_width=True
        )
        st.dataframe(dfc, use_container_width=True)
        csv = dfc.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "executive_summary.csv", "text/csv")
    else:
        st.info("Run QUBO, PPO, or MILP to populate the comparison.")

    explain_section(section, "Summarises outcomes and provides board-ready exports.")

# =========================== Meta-Learning Controller ===========================
st.divider()
st.header("Meta-Learning Controller")

st.markdown("""
This module allows the tool to automatically evaluate past performance and 
choose the best optimisation strategy (QUBO, PPO, or MILP) for the next run 
based on KPI deltas. It enables self-learning by observing its own outcomes.
""")

if "meta_memory" not in st.session_state:
    st.session_state["meta_memory"] = []

# Record current allocation + KPIs into memory
if st.button("Record Current KPIs to Memory"):
    rec = {
        "timestamp": datetime.now().isoformat(),
        "source": st.session_state.get("_alloc_source", "auto"),
        "kpi": kpi_from_allocation(st.session_state.ctx_alloc, st.session_state.vrp_cost)
    }
    st.session_state["meta_memory"].append(rec)
    st.success(f"Recorded allocation from {rec['source']} at {rec['timestamp']}.")

# Evaluate KPI deltas and select next strategy
if st.button("Evaluate and Select Next Strategy"):
    mem = st.session_state.get("meta_memory", [])
    if len(mem) < 2:
        st.warning("At least two KPI records are required for comparison.")
    else:
        last, prev = mem[-1]["kpi"], mem[-2]["kpi"]
        delta_margin = last["margin"] - prev["margin"]
        delta_co2 = prev["co2e"] - last["co2e"]
        delta_energy = prev["kwh"] - last["kwh"]

        st.write(f"ΔMargin: {delta_margin:,.0f}, ΔCO₂e: {delta_co2:,.1f}, ΔEnergy: {delta_energy:,.1f}")

        # Strategy selection logic
        if delta_margin > 0 and delta_co2 >= 0:
            next_strategy = "QUBO"
        elif delta_margin > 0 and delta_energy >= 0:
            next_strategy = "MILP"
        else:
            next_strategy = "PPO"

        st.session_state["meta_next"] = next_strategy
        st.success(f"Meta-Learner suggests next optimisation: **{next_strategy}**")

# Display meta-memory and trends
if st.checkbox("Show Learning Memory"):
    if st.session_state["meta_memory"]:
        dfm = pd.DataFrame([
            {"Time": r["timestamp"], "Source": r["source"], **r["kpi"]}
            for r in st.session_state["meta_memory"]
        ])
        st.dataframe(dfm, use_container_width=True)
        figm = px.line(dfm, x="Time", y=["margin","co2e","kwh"], title="Learning Trends (Margin/CO₂e/Energy)")
        st.plotly_chart(figm, use_container_width=True)
    else:
        st.info("No memory recorded yet. Record KPI snapshots after running QUBO, PPO, or MILP.")

st.caption("Autonomous decision loop: observes → compares → re-optimises → learns.")
