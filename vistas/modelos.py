import numpy as np
import pandas as pd
import streamlit as st

from src.charts import confusion_figure, importance_bars, response_figure, roc_figure
from src.format import fmt_int, fmt_num, fmt_pct
from src.modeling import MODEL_ORDER, ModelingError, metrics_at, response_curve
from src.state import get_bundle, get_dataset
from src.ui import disclaimer, page_header

df, meta = get_dataset()
page_header(
    "Módulo 2",
    "Tres modelos, un mismo examen",
    "Regresión logística, Random Forest y XGBoost aprenden con los mismos datos. Se comparan en un conjunto de prueba que no vieron. El recall va primero: es la capacidad de no dejar pasar un punto que sí está en riesgo.",
)

try:
    bundle = get_bundle(df)
except ModelingError as exc:
    st.error(str(exc))
    st.stop()

best = bundle.reports[bundle.recommended]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mejor recall", bundle.recommended)
c2.metric("Recall en su umbral", fmt_pct(best.metrics_tuned["recall"]))
c3.metric("ROC-AUC", fmt_num(best.metrics_tuned["roc_auc"], 3))
c4.metric("Umbral elegido", fmt_num(best.threshold, 2))

st.info(
    f"Decir siempre «sin riesgo» acertaría el {fmt_pct(1 - bundle.test_positive_rate)} de la prueba "
    f"y tendría recall de 0 %. Esa exactitud no sirve para prevenir. "
    f"El entrenamiento usa {fmt_int(bundle.n_train)} filas, la validación {fmt_int(bundle.n_val)} "
    f"y la prueba {fmt_int(bundle.n_test)}. La clase minoritaria se reequilibra con pesos: "
    "class_weight en la logística y en el bosque, y scale_pos_weight en XGBoost."
)

def metric_table(attr: str) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        report = bundle.reports[name]
        stats = getattr(report, attr)
        rows.append(
            {
                "Modelo": name,
                "Accuracy": fmt_pct(stats["accuracy"]),
                "Precision": fmt_pct(stats["precision"]),
                "Recall": fmt_pct(stats["recall"]),
                "F1": fmt_pct(stats["f1"]),
                "ROC-AUC": fmt_num(stats["roc_auc"], 3),
                "Umbral": "0,50" if attr == "metrics_default" else fmt_num(report.threshold, 2),
            }
        )
    return pd.DataFrame(rows)


st.subheader("Métricas con umbral 0,50")
st.dataframe(metric_table("metrics_default"), hide_index=True)
st.caption("Misma regla para los tres: se declara riesgo si la probabilidad pasa de 50 %.")

st.subheader("Punto de operación orientado al recall")
st.markdown(
    "El umbral de cada modelo se eligió en validación maximizando F2, que premia el recall sin soltar del todo la precisión. "
    "Los números de abajo ya son del conjunto de prueba."
)
st.dataframe(metric_table("metrics_tuned"), hide_index=True)

st.subheader("Matrices de confusión")
st.markdown(
    "La casilla que más importa está abajo a la izquierda: riesgo real que el modelo clasificó como sin riesgo. Esos son los falsos negativos."
)
cols = st.columns(3)
for column, name in zip(cols, MODEL_ORDER):
    with column:
        matrix = bundle.reports[name].metrics_default["matriz"]
        st.plotly_chart(confusion_figure(matrix, name))
        fn = int(matrix[1, 0])
        fp = int(matrix[0, 1])
        st.caption(f"Falsos negativos: {fn}. Falsos positivos: {fp}.")

st.plotly_chart(roc_figure(bundle.y_test, bundle.proba_test))

st.subheader("Explorar el umbral")
choice = st.selectbox("Modelo", MODEL_ORDER, index=MODEL_ORDER.index(bundle.recommended))
report = bundle.reports[choice]
threshold = st.slider(
    "Umbral de probabilidad",
    min_value=0.05,
    max_value=0.90,
    value=float(report.threshold),
    step=0.01,
    help="Bajar el umbral detecta más puntos en riesgo y también aumenta las alertas de más.",
)
live = metrics_at(bundle.y_test, bundle.proba_test[choice], threshold)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", fmt_pct(live["accuracy"]))
m2.metric("Precision", fmt_pct(live["precision"]))
m3.metric("Recall", fmt_pct(live["recall"]))
m4.metric("F1", fmt_pct(live["f1"]))
m5.metric("F2", fmt_pct(live["f2"]))
st.plotly_chart(confusion_figure(live["matriz"], f"{choice} en {fmt_num(threshold, 2)}"))
fn = int(live["matriz"][1, 0])
st.caption(
    f"Con este umbral quedan {fn} falsos negativos en la prueba. "
    "El recorrido usa la prueba solo para mirar; el umbral recomendado no se recalcula aquí."
)
gap = report.metrics_train["recall"] - report.metrics_default["recall"]
if gap > 0.08:
    st.caption(
        f"El recall de entrenamiento ({fmt_pct(report.metrics_train['recall'])}) supera al de prueba "
        f"({fmt_pct(report.metrics_default['recall'])}). Esa diferencia es memoria del modelo, no desempeño real."
    )

st.subheader("Qué mira cada modelo")
imp1, imp2 = st.columns(2)
with imp1:
    st.plotly_chart(
        importance_bars(
            bundle.reports["Regresión logística"].importances,
            signed=True,
            title="Regresión logística",
        ),
    )
    st.caption("Un coeficiente positivo empuja hacia riesgo. Si dos variables van de la mano, el coeficiente se reparte entre ellas.")
with imp2:
    forest_or_boost = st.radio("Árboles", ["Random Forest", "XGBoost"], horizontal=True)
    st.plotly_chart(
        importance_bars(
            bundle.reports[forest_or_boost].importances,
            signed=False,
            title=forest_or_boost,
        ),
    )
    st.caption("La importancia del árbol dice cuánto usa la variable. No dice si sube o baja el riesgo.")

st.subheader("Cómo responde el modelo recomendado")
st.markdown(f"Curvas del modelo **{bundle.recommended}**, moviendo una variable y dejando el resto como en la prueba.")
g1, g2, g3 = st.columns(3)
curves = [
    ("distancia_canada_m", np.linspace(5, 400, 18)),
    ("precipitacion_1h_mm", np.linspace(0, 60, 18)),
    ("diferencia_elevacion_m", np.linspace(0.5, 25, 18)),
]
for column, grid, host in zip([c[0] for c in curves], [c[1] for c in curves], (g1, g2, g3)):
    with host:
        values = response_curve(bundle, bundle.recommended, column, grid)
        st.plotly_chart(response_figure(grid, values, column, bundle.recommended))

disclaimer()
