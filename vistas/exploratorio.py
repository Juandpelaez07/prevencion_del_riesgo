import streamlit as st

from src.charts import (
    binned_bars,
    box_by_class,
    category_bars,
    correlation_bars,
    correlation_heatmap,
    distribution,
    risk_donut,
    scatter_risk,
)
from src.format import fmt_int, fmt_pct
from src.insights import (
    binned_risk,
    category_risk,
    correlation_with_target,
    highlights,
    outlier_table,
)
from src.mapping import MAP_IMAGE, folium_map
from src.schema import CATEGORICAL, FEATURES, label
from src.state import get_dataset
from src.ui import disclaimer, page_header

df, meta = get_dataset()
page_header(
    "Módulo 1",
    "Análisis exploratorio",
    "Antes del modelo, conviene ver cómo se reparte el riesgo: quién vive cerca de la Q. La bermejala, qué tan alto está el punto y cuánta agua cayó.",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Puntos", fmt_int(len(df)))
c2.metric("En riesgo", fmt_int(int(df["riesgo"].sum())))
c3.metric("Proporción", fmt_pct(df["riesgo"].mean()))
c4.metric("Distancia mediana", f"{fmt_int(df['distancia_canada_m'].median())} m")

if meta.get("coordenadas") == "ilustrativas":
    st.caption(
        "Este archivo no trae coordenadas. Los puntos del mapa se ubican a la distancia indicada respecto a la Q. La bermejala, solo para poder verlos."
    )

left, right = st.columns([1, 1.35])
with left:
    st.plotly_chart(risk_donut(df))
with right:
    for line in highlights(df):
        st.markdown(f"- {line}")

st.subheader("Distribución de variables")
variable = st.selectbox(
    "Variable",
    FEATURES,
    format_func=label,
    index=FEATURES.index("distancia_canada_m"),
)
st.plotly_chart(distribution(df, variable))

st.subheader("Correlaciones")
corr = correlation_with_target(df)
heat, bars = st.columns([1.15, 0.85])
with heat:
    st.plotly_chart(correlation_heatmap(df))
with bars:
    st.plotly_chart(correlation_bars(corr))
st.caption(
    "La correlación mide asociación lineal, no causa. Lluvia, humedad y cercanía al cauce suelen subir juntas con el riesgo; la distancia y la vegetación suelen bajar."
)

cat1, cat2 = st.columns(2)
with cat1:
    st.plotly_chart(category_bars(category_risk(df, "estado_drenaje"), "estado_drenaje"))
with cat2:
    st.plotly_chart(category_bars(category_risk(df, "tipo_uso_suelo"), "tipo_uso_suelo"))

st.subheader("Valores atípicos")
st.markdown(
    "Se marcan con la regla del rango intercuartílico: por fuera de Q1 − 1,5·IQR y Q3 + 1,5·IQR. "
    "En la lluvia, esos extremos no se botan. Un aguacero raro es precisamente una de las señales de riesgo."
)
table = outlier_table(df)
st.dataframe(table, hide_index=True)
outlier_var = st.selectbox(
    "Ver la caja de",
    [col for col in FEATURES if col not in CATEGORICAL],
    format_func=label,
    index=5,
)
st.plotly_chart(box_by_class(df, outlier_var))

st.subheader("Mapa")
if MAP_IMAGE.exists():
    st.image(str(MAP_IMAGE), width="stretch")
st.markdown(
    "La franja de 30 m es el borde inmediato del cauce. La de 80 m cubre la cuadra que todavía siente la Q. La bermejala. "
    "Los puntos verdes no tienen riesgo; los rojos sí."
)
try:
    from streamlit_folium import st_folium

    st_folium(
        folium_map(df),
        height=560,
        use_container_width=True,
        returned_objects=[],
        key="mapa_eda",
    )
except Exception:
    st.info("El mapa interactivo no está disponible en este momento. El plano de referencia sigue arriba.")

st.subheader("Precipitación")
st.markdown(
    "La lluvia entra en cinco ventanas, de la última hora a los siete días. "
    "La hora pesa en la creciente súbita; la semana pesa en qué tan lleno está ya el suelo."
)
rain_specs = [
    ("precipitacion_1h_mm", [0, 2, 5, 10, 20, 40, 130], ["0–2", "2–5", "5–10", "10–20", "20–40", "40+"]),
    ("precipitacion_24h_mm", [0, 5, 15, 30, 50, 80, 260], ["0–5", "5–15", "15–30", "30–50", "50–80", "80+"]),
    ("precipitacion_7d_mm", [0, 15, 30, 60, 100, 160, 500], ["0–15", "15–30", "30–60", "60–100", "100–160", "160+"]),
]
rain_cols = st.columns(3)
for column, edges, names, col in zip(
    [item[0] for item in rain_specs],
    [item[1] for item in rain_specs],
    [item[2] for item in rain_specs],
    rain_cols,
):
    with col:
        summary = binned_risk(df, column, edges, names)
        st.plotly_chart(binned_bars(summary, label(column)))

st.plotly_chart(
    scatter_risk(df, "precipitacion_7d_mm", "humedad_suelo_pct", "Lluvia de la semana y humedad del suelo"),
)

st.subheader("Distancia a la Q. La bermejala y riesgo")
dist_bins = binned_risk(
    df,
    "distancia_canada_m",
    [0, 20, 40, 80, 150, 300, 900],
    ["0–20 m", "20–40", "40–80", "80–150", "150–300", "300+"],
)
d1, d2 = st.columns(2)
with d1:
    st.plotly_chart(binned_bars(dist_bins, "Proporción en riesgo según la distancia"))
with d2:
    st.plotly_chart(
        scatter_risk(df, "distancia_canada_m", "diferencia_elevacion_m", "Cercanía y altura sobre el cauce"),
    )

st.subheader("Elevación y riesgo")
e1, e2 = st.columns(2)
with e1:
    elev_bins = binned_risk(
        df,
        "elevacion_msnm",
        [1470, 1510, 1540, 1570, 1600, 1640, 1720],
        ["<1510", "1510–1540", "1540–1570", "1570–1600", "1600–1640", "1640+"],
    )
    st.plotly_chart(binned_bars(elev_bins, "Elevación del punto"))
with e2:
    diff_bins = binned_risk(
        df,
        "diferencia_elevacion_m",
        [0, 2, 5, 8, 12, 20, 50],
        ["0–2 m", "2–5", "5–8", "8–12", "12–20", "20+"],
    )
    st.plotly_chart(binned_bars(diff_bins, "Metros por encima del cauce"))
st.caption(
    "La elevación absoluta dice en qué parte de Aranjuez está el punto: arriba en la ladera o abajo, hacia el río. "
    "La diferencia de elevación dice si un desborde corto lo alcanza. Las dos importan, y no dicen lo mismo."
)

disclaimer()
