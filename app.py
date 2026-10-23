# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:56:55 2026

@author: pacol
"""

import numpy as np
import plotly.graph_objects as go
from scipy.stats import maxwell
import streamlit as st

st.set_page_config(
    page_title="Distribución de Maxwell-Boltzmann",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Distribución de Velocidades de Maxwell-Boltzmann")
st.markdown("Simulador interactivo para el **Tema 1: Teoría Cinética de Gases**.")

# Constante universal de los gases ideales (J / (mol K))
R = 8.314462618

st.sidebar.header("Parámetros del Sistema")

gases = {
    "Hidrógeno (H₂)": 2.016,
    "Helio (He)": 4.003,
    "Nitrógeno (N₂)": 28.013,
    "Oxígeno (O₂)": 31.999,
    "Argón (Ar)": 39.948,
    "Xenón (Xe)": 131.29,
}

gas_seleccionado = st.sidebar.selectbox("Selecciona un gas:", list(gases.keys()))
M_g_mol = gases[gas_seleccionado]
M = M_g_mol / 1000.0  # Conversión a kg/mol

T = st.sidebar.slider(
    "Temperatura T (K):", min_value=100, max_value=1500, value=300, step=10
)
v_corte = st.sidebar.slider(
    "Velocidad de corte v_corte (m/s):",
    min_value=100,
    max_value=2500,
    value=800,
    step=25,
)

# Velocidades características
v_p = np.sqrt(2 * R * T / M)
v_med = np.sqrt(8 * R * T / (np.pi * M))
v_rms = np.sqrt(3 * R * T / M)

# Parámetro de escala para scipy.stats.maxwell: scale = sqrt(R*T / M)
scale = np.sqrt(R * T / M)

# Fracción de moléculas por encima de v_corte (sf = Survival Function = 1 - cdf)
fraccion_supera = maxwell.sf(v_corte, scale=scale) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Velocidad más probable (v_p)", f"{v_p:.1f} m/s")
col2.metric("Velocidad media (⟨v⟩)", f"{v_med:.1f} m/s")
col3.metric("Velocidad cuadrática media (v_rms)", f"{v_rms:.1f} m/s")
col4.metric(f"% Moléculas con v > {v_corte} m/s", f"{fraccion_supera:.2f}%")

# Eje de velocidades
v_max = max(3000.0, v_rms * 2.5)
v = np.linspace(0, v_max, 600)
f_v = maxwell.pdf(v, scale=scale)

fig = go.Figure()

# Curva principal f(v)
fig.add_trace(
    go.Scatter(
        x=v, y=f_v, mode="lines", name="f(v)", line=dict(color="#1f77b4", width=3)
    )
)

# Área de moléculas activadas / efusión (v > v_corte)
v_corte_arr = np.linspace(v_corte, v_max, 300)
f_v_corte = maxwell.pdf(v_corte_arr, scale=scale)

fig.add_trace(
    go.Scatter(
        x=np.concatenate([[v_corte], v_corte_arr, [v_max]]),
        y=np.concatenate([[0], f_v_corte, [0]]),
        fill="tozeroy",
        fillcolor="rgba(255, 127, 14, 0.3)",
        line=dict(color="rgba(255, 127, 14, 0)"),
        name=f"v > {v_corte} m/s",
        hoverinfo="skip",
    )
)

# Marcadores de velocidades características
fig.add_vline(
    x=v_p,
    line_dash="dash",
    line_color="#2ca02c",
    annotation_text=f"v_p ({v_p:.0f})",
)
fig.add_vline(
    x=v_med,
    line_dash="dash",
    line_color="#d62728",
    annotation_text=f"⟨v⟩ ({v_med:.0f})",
)
fig.add_vline(
    x=v_rms,
    line_dash="dash",
    line_color="#9467bd",
    annotation_text=f"v_rms ({v_rms:.0f})",
)

fig.update_layout(
    xaxis_title="Velocidad molecular v (m/s)",
    yaxis_title="Densidad de probabilidad f(v) (s/m)",
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(x=0.7, y=0.95),
    height=520,
)

st.plotly_chart(fig, use_container_width=True)