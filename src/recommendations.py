"""Niveles de riesgo y recomendaciones de prevención."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.format import fmt_measure
from src.schema import FEATURES


@dataclass(frozen=True)
class Level:
    key: str
    name: str
    low: float
    high: float
    title: str
    summary: str
    color: str
    ink: str
    actions: tuple[str, ...]


LEVELS: tuple[Level, ...] = (
    Level(
        "bajo",
        "Bajo",
        0.0,
        0.25,
        "Monitoreo normal",
        "Con las condiciones ingresadas, el punto no concentra una probabilidad alta de afectación.",
        "#2f7d4a",
        "#f4faf6",
        (
            "Mantener la observación rutinaria del cauce y de los sumideros de la cuadra.",
            "No ocupar el retiro de la Q. La bermejala con enseres, escombros o residuos.",
            "Tener identificada la ruta de evacuación del sector, aunque hoy no haya alerta.",
        ),
    ),
    Level(
        "medio",
        "Medio",
        0.25,
        0.50,
        "Seguimiento de lluvias",
        "El punto merece vigilancia: un aumento de la lluvia puede cambiar el resultado en poco tiempo.",
        "#b5810b",
        "#fffaf0",
        (
            "Revisar el pronóstico y los avisos de SIATA durante las próximas horas.",
            "Evitar dejar vehículos, mercados o materiales en la margen de la Q. La bermejala.",
            "Confirmar que los drenajes y las rejillas de la cuadra estén libres.",
            "Acordar un punto de encuentro si la lluvia se intensifica.",
        ),
    ),
    Level(
        "alto",
        "Alto",
        0.50,
        0.75,
        "Evitar actividades cercanas al cauce",
        "La combinación de terreno y lluvia deja el punto en una franja que conviene abandonar mientras dure el evento.",
        "#d4652f",
        "#fff8f4",
        (
            "Suspender permanencia, trabajo o juego en el retiro, los pasos bajos y el borde del cauce.",
            "Subir a un piso alto o desplazarse fuera de la franja próxima a la Q. La bermejala.",
            "No cruzar el cauce ni calles con agua en movimiento.",
            "Atender las indicaciones de los líderes del sector y del DAGRD.",
        ),
    ),
    Level(
        "muy_alto",
        "Muy alto",
        0.75,
        1.01,
        "Priorizar medidas preventivas y atención a alertas oficiales",
        "El punto queda primero en la lista: hay que actuar antes de que el nivel de la Q. La bermejala suba.",
        "#b42318",
        "#fff6f5",
        (
            "Alejarse del cauce antes de que suba el nivel. No esperar a ver el agua en la puerta.",
            "Seguir únicamente alertas oficiales del DAGRD y de SIATA.",
            "Activar la ruta de evacuación y ayudar a quienes tengan movilidad reducida.",
            "No regresar a la margen hasta que la autoridad lo indique.",
        ),
    ),
)


def classify(probability: float) -> Level:
    for level in LEVELS:
        if level.low <= probability < level.high:
            return level
    return LEVELS[-1]


def contextual_notes(row: pd.Series) -> list[str]:
    notes = []
    if float(row["precipitacion_1h_mm"]) >= 20:
        notes.append(
            "La lluvia de la última hora ya es intensa. En la Q. La bermejala, la creciente puede llegar en minutos."
        )
    if float(row["precipitacion_7d_mm"]) >= 90 and float(row["precipitacion_1h_mm"]) < 20:
        notes.append(
            "Aunque esta hora no sea extrema, la lluvia de la semana ya pudo saturar el suelo."
        )
    if float(row["distancia_canada_m"]) <= 30:
        notes.append(
            "El punto está en los primeros 30 m. Conviene salir de esa franja antes de que el nivel suba."
        )
    if float(row["diferencia_elevacion_m"]) <= 3:
        notes.append(
            "La diferencia de altura frente al cauce es pequeña. Una creciente moderada puede alcanzar el punto."
        )
    if row["estado_drenaje"] in {"Deficiente", "Colapsado"}:
        notes.append(
            "El drenaje no está en condición de evacuar un aguacero. No cuente con la alcantarilla."
        )
    if int(row["historial_inundacion_eventos"]) >= 2:
        notes.append(
            "Este perfil acumula antecedentes. Use la ruta que el sector ya haya reconocido, sin improvisar un cruce."
        )
    if float(row["superficie_impermeable_pct"]) >= 75:
        notes.append(
            "La superficie impermeable acelera el escurrimiento: el agua llega más rápido al cauce."
        )
    if float(row["humedad_suelo_pct"]) >= 75:
        notes.append("El suelo ya está húmedo. Otra lluvia se convierte casi toda en escorrentía.")
    if float(row["cobertura_vegetal_pct"]) <= 12:
        notes.append("Hay muy poca vegetación que retenga suelo o frene el agua en la margen.")
    if float(row["pendiente_pct"]) >= 25 and float(row["distancia_canada_m"]) <= 80:
        notes.append(
            "La pendiente es fuerte cerca del cauce. El agua y el material que arrastra ganan velocidad."
        )
    return notes


def factor_sentence(column: str, value, reference, delta: float) -> str:
    value_txt = fmt_measure(column, value)
    ref_txt = fmt_measure(column, reference)
    direction = "empuja la probabilidad hacia arriba" if delta > 0 else "la contiene"
    specifics = {
        "distancia_canada_m": f"La Q. La bermejala queda a {value_txt}. Un punto típico del dataset está a {ref_txt}.",
        "distancia_drenaje_m": f"El drenaje más cercano está a {value_txt} (típico: {ref_txt}).",
        "elevacion_msnm": f"La elevación del punto es {value_txt}; la mediana del dataset es {ref_txt}.",
        "diferencia_elevacion_m": f"El punto está {value_txt} por encima del cauce (típico: {ref_txt}).",
        "pendiente_pct": f"La pendiente es {value_txt}, frente a una mediana de {ref_txt}.",
        "precipitacion_1h_mm": f"En la última hora van {value_txt}. Lo habitual en el dataset es {ref_txt}.",
        "precipitacion_6h_mm": f"La lluvia de 6 horas es {value_txt} (típico: {ref_txt}).",
        "precipitacion_24h_mm": f"La lluvia de 24 horas es {value_txt} (típico: {ref_txt}).",
        "precipitacion_3d_mm": f"En tres días se acumulan {value_txt} (típico: {ref_txt}).",
        "precipitacion_7d_mm": f"La semana acumula {value_txt} (típico: {ref_txt}).",
        "humedad_suelo_pct": f"La humedad del suelo está en {value_txt} (típico: {ref_txt}).",
        "cobertura_vegetal_pct": f"La cobertura vegetal es {value_txt} (típico: {ref_txt}).",
        "superficie_impermeable_pct": f"La superficie impermeable es {value_txt} (típico: {ref_txt}).",
        "densidad_construccion_pct": f"La densidad de construcción es {value_txt} (típico: {ref_txt}).",
        "tipo_uso_suelo": f"El uso del suelo es {value_txt}. Lo más frecuente en el dataset es {ref_txt}.",
        "estado_drenaje": f"El drenaje está en estado {value_txt}. Lo más frecuente es {ref_txt}.",
        "historial_inundacion_eventos": f"El historial registra {value_txt}. La mediana del dataset es {ref_txt}.",
    }
    body = specifics.get(column, f"{column}: {value_txt}.")
    return f"{body} Eso {direction}."


def row_from_mapping(values: dict) -> pd.DataFrame:
    return pd.DataFrame([{col: values[col] for col in FEATURES}])
