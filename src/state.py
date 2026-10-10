"""Dataset activo y modelos en caché."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.dataset import default_source, load_table
from src.modeling import Bundle, train_all


@st.cache_data(show_spinner=False)
def _load_default():
    frame, meta = load_table(default_source())
    meta = dict(meta)
    meta["origen"] = "Dataset de demostración"
    return frame, meta


def get_dataset() -> tuple[pd.DataFrame, dict]:
    if "df_override" in st.session_state:
        return st.session_state["df_override"], st.session_state["df_meta"]
    return _load_default()


@st.cache_resource(show_spinner="Entrenando regresión logística, Random Forest y XGBoost…")
def _train(payload: str) -> Bundle:
    frame = pd.read_csv(io.StringIO(payload))
    return train_all(frame)


def get_bundle(df: pd.DataFrame) -> Bundle:
    return _train(df.to_csv(index=False))
