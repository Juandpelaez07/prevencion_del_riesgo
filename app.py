import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DatasetError
from src.format import fmt_int, fmt_pct
from src.state import get_dataset
from src.ui import inject_css

st.set_page_config(
    page_title="Q. La bermejala Aranjuez — Prevención del riesgo",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

inicio = st.Page(ROOT / "vistas" / "inicio.py", title="Inicio", icon="🏠", default=True)
exploratorio = st.Page(ROOT / "vistas" / "exploratorio.py", title="Exploratorio", icon="📊")
modelos = st.Page(ROOT / "vistas" / "modelos.py", title="Modelos", icon="🤖")
prediccion = st.Page(ROOT / "vistas" / "prediccion.py", title="Predicción", icon="📍")
recomendaciones = st.Page(ROOT / "vistas" / "recomendaciones.py", title="Recomendaciones", icon="💡")

st.sidebar.markdown(
    """
    <div class="brand">
      <p class="brand-kicker">Comuna 4 · Medellín</p>
      <p class="brand-title">Q. La bermejala<br>Aranjuez</p>
      <p class="brand-sub">Prevención del riesgo</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    frame, meta = get_dataset()
except DatasetError as exc:
    st.sidebar.error(str(exc))
    frame, meta = None, None

if frame is not None:
    st.sidebar.caption(f"{fmt_int(len(frame))} registros · {fmt_pct(meta['tasa_riesgo'])} en riesgo")
    st.sidebar.caption(meta.get("origen", ""))
    if meta.get("coordenadas") == "ilustrativas":
        st.sidebar.caption("Coordenadas ilustrativas, reconstruidas con la distancia a la Q. La bermejala.")
    if meta.get("filas_descartadas"):
        st.sidebar.caption(f"Se omitieron {fmt_int(meta['filas_descartadas'])} filas incompletas.")

st.navigation(
    {
        "Lectura": [inicio, exploratorio],
        "Decisión": [modelos, prediccion, recomendaciones],
    }
).run()
