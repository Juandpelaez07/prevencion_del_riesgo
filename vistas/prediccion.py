import pandas as pd
import streamlit as st

from src.format import fmt_num, fmt_pct
from src.geography import SECTOR_BY_NAME, SECTORS, channel_length, point_at, sector_profile
from src.modeling import MODEL_ORDER, ModelingError, local_factors, predict_proba
from src.recommendations import classify, contextual_notes, factor_sentence, row_from_mapping
from src.schema import ESTADO_DRENAJE, FEATURES, PRECIP_CHAIN, USO_SUELO
from src.state import get_bundle, get_dataset
from src.ui import disclaimer, page_header, risk_hero

RAIN = {
    "precipitacion_1h_mm": 1.0,
    "precipitacion_6h_mm": 2.0,
    "precipitacion_24h_mm": 6.0,
    "precipitacion_3d_mm": 10.0,
    "precipitacion_7d_mm": 18.0,
    "humedad_suelo_pct": 30.0,
}

SCENARIOS = {
    "Brasilia en calma": {
        "sector": "Brasilia",
        "precipitacion_1h_mm": 1.0,
        "precipitacion_6h_mm": 2.0,
        "precipitacion_24h_mm": 6.0,
        "precipitacion_3d_mm": 10.0,
        "precipitacion_7d_mm": 18.0,
        "humedad_suelo_pct": 30.0,
    },
    "Lluvia breve en la ladera": {
        "sector": "Las Esmeraldas",
        "precipitacion_1h_mm": 4.0,
        "precipitacion_6h_mm": 8.0,
        "precipitacion_24h_mm": 14.0,
        "precipitacion_3d_mm": 22.0,
        "precipitacion_7d_mm": 36.0,
        "humedad_suelo_pct": 36.0,
    },
    "Bermejal sin lluvia": {
        "sector": "Bermejal",
        "precipitacion_1h_mm": 0.0,
        "precipitacion_6h_mm": 1.0,
        "precipitacion_24h_mm": 3.0,
        "precipitacion_3d_mm": 7.0,
        "precipitacion_7d_mm": 12.0,
        "humedad_suelo_pct": 28.0,
    },
    "Aguacero en la margen": {
        "sector": "Bermejal",
        "precipitacion_1h_mm": 42.0,
        "precipitacion_6h_mm": 58.0,
        "precipitacion_24h_mm": 76.0,
        "precipitacion_3d_mm": 94.0,
        "precipitacion_7d_mm": 120.0,
        "humedad_suelo_pct": 84.0,
    },
}


def apply_sector(name: str, rain: dict | None = None) -> None:
    profile = sector_profile(SECTOR_BY_NAME[name])
    for column in FEATURES:
        if column in profile:
            st.session_state[column] = profile[column]
    st.session_state["sector"] = name
    if rain:
        for key, value in rain.items():
            st.session_state[key] = value


def ensure_inputs(default_model: str) -> None:
    if "distancia_canada_m" not in st.session_state:
        apply_sector("Brasilia", RAIN)
    if "modelo_pred" not in st.session_state:
        st.session_state["modelo_pred"] = default_model


def current_row() -> pd.DataFrame:
    values = {column: st.session_state[column] for column in FEATURES}
    return row_from_mapping(values)


def point_for_sector(name: str, distancia: float) -> tuple[float, float]:
    sector = SECTOR_BY_NAME[name]
    along = sector.along_frac * channel_length()
    sign = 1 if sector.offset_m >= 0 else -1
    return point_at(along, sign * float(distancia))


def store_result(bundle) -> None:
    row = current_row()
    name = st.session_state["modelo_pred"]
    probability = predict_proba(bundle, name, row)
    _, increases, decreases = local_factors(bundle, name, row)
    lat, lon = point_for_sector(st.session_state["sector"], row.iloc[0]["distancia_canada_m"])
    st.session_state["resultado"] = {
        "probabilidad": probability,
        "modelo": name,
        "umbral": bundle.reports[name].threshold,
        "suben": increases,
        "bajan": decreases,
        "notas": contextual_notes(row.iloc[0]),
        "fila": row.iloc[0].to_dict(),
        "latitud": lat,
        "longitud": lon,
        "sector": st.session_state["sector"],
        "todos": {model: predict_proba(bundle, model, row) for model in MODEL_ORDER},
    }


df, _meta = get_dataset()
page_header(
    "Módulo 3",
    "Consultar un punto",
    "Elija un sector de Aranjuez y ajuste la lluvia, el terreno y la urbanización. El resultado es la probabilidad de que ese punto esté en riesgo, y la recomendación que le corresponde.",
)

try:
    bundle = get_bundle(df)
except ModelingError as exc:
    st.error(str(exc))
    st.stop()

ensure_inputs(bundle.recommended)
pending = st.session_state.pop("pending_apply", None)
if pending:
    st.session_state["modelo_pred"] = pending.get("modelo", st.session_state["modelo_pred"])
    apply_sector(pending["sector"], pending["rain"])
    if pending.get("predict"):
        store_result(bundle)

st.selectbox("Ubicación en Aranjuez", [sector.nombre for sector in SECTORS], key="sector")
st.caption(SECTOR_BY_NAME[st.session_state["sector"]].nota)
st.caption("Los valores del sector son supuestos de este ejercicio. No son la calificación oficial del barrio.")
if st.button("Cargar condiciones del sector", type="secondary"):
    st.session_state["pending_apply"] = {
        "sector": st.session_state["sector"],
        "rain": {key: st.session_state[key] for key in RAIN},
        "predict": False,
        "modelo": st.session_state["modelo_pred"],
    }
    st.rerun()

st.markdown("**Escenarios de ejemplo**")
scenario_cols = st.columns(4)
for column, (title, rain) in zip(scenario_cols, SCENARIOS.items()):
    with column:
        if st.button(title, width="stretch"):
            st.session_state["pending_apply"] = {
                "sector": rain["sector"],
                "rain": rain,
                "predict": True,
                "modelo": bundle.recommended,
            }
            st.rerun()

with st.form("consulta"):
    terreno, clima = st.columns(2)
    with terreno:
        st.markdown("**Q. La bermejala, elevación y pendiente**")
        st.slider("Distancia a la Q. La bermejala (m)", 0.0, 800.0, key="distancia_canada_m")
        st.slider("Distancia al drenaje (m)", 1.0, 400.0, key="distancia_drenaje_m")
        st.slider("Elevación (msnm)", 1475.0, 1680.0, key="elevacion_msnm")
        st.slider("Diferencia de elevación frente al cauce (m)", 0.0, 40.0, key="diferencia_elevacion_m")
        st.slider("Pendiente (%)", 0.0, 55.0, key="pendiente_pct")
        st.selectbox("Estado del drenaje", ESTADO_DRENAJE, key="estado_drenaje")
    with clima:
        st.markdown("**Lluvia y urbanización**")
        st.slider("Lluvia última hora (mm)", 0.0, 120.0, key="precipitacion_1h_mm")
        st.slider("Lluvia 24 horas (mm)", 0.0, 180.0, key="precipitacion_24h_mm")
        st.slider("Lluvia 7 días (mm)", 0.0, 280.0, key="precipitacion_7d_mm")
        st.slider("Humedad del suelo (%)", 0.0, 100.0, key="humedad_suelo_pct")
        st.selectbox("Uso del suelo", USO_SUELO, key="tipo_uso_suelo")
        st.slider("Superficie impermeable (%)", 0.0, 100.0, key="superficie_impermeable_pct")
        st.slider("Densidad de construcción (%)", 0.0, 100.0, key="densidad_construccion_pct")
        st.slider("Cobertura vegetal (%)", 0.0, 100.0, key="cobertura_vegetal_pct")
        st.slider("Historial de inundación (eventos)", 0, 12, key="historial_inundacion_eventos")
    with st.expander("Lluvia intermedia"):
        st.slider("Lluvia 6 horas (mm)", 0.0, 150.0, key="precipitacion_6h_mm")
        st.slider("Lluvia 3 días (mm)", 0.0, 220.0, key="precipitacion_3d_mm")
        st.caption("Si no las conoce, deje valores entre la hora y las 24 h, y entre las 24 h y los 7 días.")
    st.selectbox("Modelo", MODEL_ORDER, key="modelo_pred")
    submitted = st.form_submit_button("Calcular riesgo", type="primary")

if submitted:
    store_result(bundle)

ordered = [st.session_state[col] for col in PRECIP_CHAIN]
if any(left > right + 0.05 for left, right in zip(ordered, ordered[1:])):
    st.warning(
        "La lluvia de una ventana corta no debería superar la de una ventana más larga. "
        "Revise las horas y los días antes de leer el resultado como un evento real."
    )

resultado = st.session_state.get("resultado")
if not resultado:
    st.markdown(
        "Calcule con las condiciones del sector o abra un escenario. "
        "Brasilia en calma y el aguacero sobre Bermejal quedan en extremos distintos: "
        "la margen sigue expuesta aunque esa hora no llueva."
    )
    st.stop()

level = classify(resultado["probabilidad"])
risk_hero(resultado["probabilidad"], level)

st.markdown("")
actions, context = st.columns([1.15, 0.85])
with actions:
    st.markdown(f"#### {level.title}")
    for action in level.actions:
        st.markdown(f"- {action}")
with context:
    if resultado["notas"]:
        st.markdown("#### En este punto")
        for note in resultado["notas"]:
            st.markdown(f"- {note}")
    else:
        st.markdown("#### En este punto")
        st.markdown("Ninguna condición suelta cambia el consejo general de este nivel.")

st.subheader("Por qué salió así")
st.caption(
    "Cada frase compara el valor ingresado con un punto típico del dataset y mide cuánto mueve la probabilidad."
)
up_col, down_col = st.columns(2)
with up_col:
    st.markdown("**Empujan el riesgo**")
    if resultado["suben"]:
        for column, delta, value, reference in resultado["suben"]:
            st.markdown(f"- {factor_sentence(column, value, reference, delta)} ({fmt_num(delta * 100, 1)} puntos)")
    else:
        st.markdown("Ninguna variable se aleja del caso típico hacia más riesgo.")
with down_col:
    st.markdown("**Lo contienen**")
    if resultado["bajan"]:
        for column, delta, value, reference in resultado["bajan"]:
            st.markdown(f"- {factor_sentence(column, value, reference, delta)} ({fmt_num(delta * 100, 1)} puntos)")
    else:
        st.markdown("Ninguna variable está conteniendo el riesgo de forma clara.")

st.subheader("Los tres modelos en este punto")
compare = pd.DataFrame(
    {
        "Modelo": list(resultado["todos"]),
        "Probabilidad": [fmt_pct(value) for value in resultado["todos"].values()],
        "Nivel": [classify(value).name for value in resultado["todos"].values()],
    }
)
st.dataframe(compare, hide_index=True)
report = bundle.reports[resultado["modelo"]]
st.caption(
    f"La ficha usa {resultado['modelo']}. Su umbral de trabajo, elegido en validación, es {fmt_num(report.threshold, 2)}. "
    f"La clasificación de cuatro colores sigue las franjas 25 %, 50 % y 75 %, no ese umbral."
)

st.subheader("Dónde queda el punto")
st.caption(
    f"{resultado['sector']}, a {fmt_num(resultado['fila']['distancia_canada_m'], 0)} m del eje de la Q. La bermejala de estudio."
)
try:
    from streamlit_folium import st_folium

    from src.mapping import folium_map

    st_folium(
        folium_map(
            df,
            point={
                "latitud": resultado["latitud"],
                "longitud": resultado["longitud"],
                "sector": resultado["sector"],
            },
        ),
        height=460,
        use_container_width=True,
        returned_objects=[],
        key="mapa_pred",
    )
except Exception:
    st.caption("El mapa interactivo no pudo dibujarse en esta sesión.")

disclaimer()
