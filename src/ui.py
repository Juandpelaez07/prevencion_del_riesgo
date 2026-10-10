"""Estilo compartido de la aplicación."""

from __future__ import annotations

import streamlit as st

from src.format import fmt_num
from src.recommendations import Level


def inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,650&family=Outfit:wght@340;420;560&display=swap');

          html, body, [class*="css"] {
            font-family: Outfit, "Segoe UI", sans-serif;
          }
          .stApp {
            background:
              radial-gradient(1200px 500px at 10% -10%, rgba(31,107,74,0.08), transparent 50%),
              #f3f0e8;
            color: #17241c;
          }
          h1, h2, h3, .brand-title, .prob, .level-name {
            font-family: Fraunces, Georgia, "Palatino Linotype", serif !important;
            letter-spacing: -0.03em;
            font-weight: 560;
          }
          header[data-testid="stHeader"] { background: transparent; }
          footer { visibility: hidden; }
          [data-testid="stSidebar"] {
            background: #14241c;
            border-right: 1px solid #0d1813;
          }
          [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] span,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] small,
          [data-testid="stSidebar"] li {
            color: #e7efe8 !important;
          }
          [data-testid="stSidebar"] a {
            border-radius: 10px;
          }
          [data-testid="stSidebar"] a[aria-current="page"] {
            background: rgba(247, 244, 238, 0.1);
            box-shadow: inset 3px 0 0 #e2c27a;
          }
          [data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 0.2rem 0.4rem 0.6rem;
          }
          .brand { padding: 0.4rem 0.2rem 0.8rem; }
          .brand-kicker, .kicker {
            font-family: Outfit, sans-serif !important;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.72rem;
            color: #8ea396;
            margin: 0 0 0.35rem;
          }
          .brand-title {
            color: #f7f4ee;
            font-size: 1.85rem;
            line-height: 0.95;
            margin: 0;
          }
          .brand-sub {
            color: #e2c27a !important;
            margin: 0.35rem 0 0;
            font-size: 0.95rem;
          }
          .page-head { margin: 0.2rem 0 1.2rem; max-width: 760px; }
          .page-head h1 {
            font-size: 3rem;
            line-height: 0.96;
            margin: 0 0 0.6rem;
            color: #17241c;
          }
          .lede {
            font-size: 1.12rem;
            line-height: 1.5;
            color: #3d5147;
            margin: 0;
          }
          .module-card, .level-card, .note-card {
            background: #fffdf8;
            border: 1px solid #e4d9c8;
            border-radius: 18px;
            padding: 1rem 1.1rem 0.9rem;
            height: 100%;
          }
          .module-card h3, .level-card h3 { margin: 0.2rem 0 0.4rem; font-size: 1.45rem; }
          .module-card p, .level-card p, .note-card p { color: #3d5147; margin: 0.2rem 0; }
          .level-card { border-top: 6px solid var(--tone); }
          [data-testid="stMetric"] {
            background: #fffdf8;
            border: 1px solid #e4d9c8;
            border-radius: 16px;
            padding: 0.7rem 0.9rem 0.5rem;
          }
          [data-testid="stMetricValue"] { font-family: Fraunces, Georgia, serif; }
          .risk-hero {
            display: grid;
            grid-template-columns: 1.1fr 1.4fr;
            gap: 1.2rem;
            background: var(--ink);
            color: #17241c;
            border-radius: 22px;
            padding: 1.3rem 1.4rem 1.2rem;
            border: 1px solid color-mix(in srgb, var(--tone) 45%, #e4d9c8);
          }
          .prob {
            font-size: 5.2rem;
            line-height: 0.85;
            margin: 0.1rem 0 0.6rem;
            color: var(--tone);
          }
          .prob span { font-size: 1.8rem; margin-left: 0.15rem; }
          .bar {
            height: 10px;
            background: rgba(23, 36, 28, 0.08);
            border-radius: 99px;
            overflow: hidden;
          }
          .bar span {
            display: block;
            height: 100%;
            background: var(--tone);
          }
          .level-name {
            font-size: 2.4rem;
            margin: 0;
            color: var(--tone);
          }
          .level-title { font-size: 1.15rem; margin: 0.25rem 0 0; color: #17241c; }
          .disclaimer {
            color: #5c6b62;
            font-size: 0.92rem;
            line-height: 1.45;
          }
          @media (max-width: 800px) {
            .page-head h1 { font-size: 2.2rem; }
            .risk-hero { grid-template-columns: 1fr; }
            .prob { font-size: 4rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, lede: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <p class="kicker">{kicker}</p>
          <h1>{title}</h1>
          <p class="lede">{lede}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_hero(probability: float, level: Level) -> None:
    pct = fmt_num(probability * 100, 0)
    width = max(0.0, min(probability * 100, 100))
    st.markdown(
        f"""
        <div class="risk-hero" style="--tone:{level.color};--ink:{level.ink}">
          <div>
            <p class="kicker">Probabilidad de riesgo</p>
            <p class="prob">{pct}<span>%</span></p>
            <div class="bar"><span style="width:{width:.1f}%"></span></div>
          </div>
          <div class="level">
            <p class="kicker">Clasificación</p>
            <p class="level-name">{level.name}</p>
            <p class="level-title">{level.title}</p>
            <p>{level.summary}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        """
        <p class="disclaimer">
        Este ejercicio usa un dataset sintético y un plano de referencia de la Q. La bermejala en Aranjuez.
        La clasificación no equivale a las categorías jurídicas de riesgo del POT y no reemplaza
        las alertas del DAGRD ni de SIATA. Ante una emergencia, la línea es 123.
        </p>
        """,
        unsafe_allow_html=True,
    )
