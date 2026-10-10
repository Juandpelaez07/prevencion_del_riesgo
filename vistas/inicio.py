import streamlit as st

from src.format import fmt_int, fmt_pct
from src.mapping import MAP_IMAGE
from src.state import get_dataset
from src.ui import disclaimer, page_header

df, meta = get_dataset()
page_header(
    "Comuna 4 · Medellín",
    "¿Este punto está en riesgo?",
    "Una lectura para la Q. La bermejala de Aranjuez: con la distancia al cauce, la lluvia y la forma del barrio, el sistema dice si el punto queda en zona de riesgo o si basta con recomendaciones de seguimiento.",
)

if MAP_IMAGE.exists():
    st.image(str(MAP_IMAGE), width="stretch")
else:
    st.info("Falta el plano de referencia. Genérelo con python scripts/generate_dataset.py.")

st.caption(
    "Plano de trabajo. El trazo resalta la Q. La bermejala de estudio dentro de Aranjuez; no es la delimitación oficial del DAP."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", fmt_int(len(df)))
c2.metric("En riesgo", fmt_pct(meta["tasa_riesgo"]))
c3.metric("Sin riesgo", fmt_pct(1 - meta["tasa_riesgo"]))
c4.metric("Variables", "17")

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        """
        <div class="note-card">
          <p class="kicker">El problema</p>
          <h3>No basta con saber que la Q. La bermejala existe</h3>
          <p>
            En Aranjuez el agua baja por la Q. La bermejala que cruza un tejido ya construido.
            Un punto puede estar a pocos metros del cauce, casi a su misma altura, con el suelo
            saturado y el drenaje sin capacidad. Otro, a la vuelta, solo necesita seguimiento.
          </p>
          <p>
            La pregunta de este ejercicio es concreta: con las condiciones de un punto,
            ¿hay riesgo o no? El dataset marca 1 cuando el punto queda en riesgo y 0 cuando no.
            Cerca del 71 % de los casos no tiene riesgo y el 29 % sí.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class="note-card">
          <p class="kicker">Cómo se lee</p>
          <h3>Primero detectar, después aconsejar</h3>
          <p>
            En un problema de riesgo, dejar pasar un punto expuesto —un falso negativo—
            pesa más que una alerta de más. Por eso los modelos se comparan sobre todo por el recall.
          </p>
          <p>
            La probabilidad se traduce en cuatro niveles: bajo, medio, alto y muy alto.
            Cada uno trae una recomendación. Ninguna sustituye una alerta oficial.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Los cuatro módulos")
modules = [
    ("01", "Análisis exploratorio", "Distribuciones, correlaciones, atípicos, lluvia, distancia, elevación y el mapa.", "vistas/exploratorio.py"),
    ("02", "Machine learning", "Regresión logística, Random Forest y XGBoost, con la matriz de confusión y el recall a la vista.", "vistas/modelos.py"),
    ("03", "Predicción", "Ubicación, lluvia, elevación, pendiente, distancia a la Q. La bermejala y urbanización.", "vistas/prediccion.py"),
    ("04", "Recomendaciones", "De monitoreo normal a atención de alertas oficiales, según el nivel.", "vistas/recomendaciones.py"),
]
cols = st.columns(2)
for i, (num, title, text, page) in enumerate(modules):
    with cols[i % 2]:
        st.markdown(
            f"""
            <div class="module-card">
              <p class="kicker">{num}</p>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(page, label=f"Abrir {title.lower()}")

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
disclaimer()
