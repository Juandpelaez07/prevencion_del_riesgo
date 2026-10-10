"""Columnas del dataset y nombres para la interfaz."""

FEATURES = [
    "distancia_canada_m",
    "distancia_drenaje_m",
    "elevacion_msnm",
    "diferencia_elevacion_m",
    "pendiente_pct",
    "precipitacion_1h_mm",
    "precipitacion_6h_mm",
    "precipitacion_24h_mm",
    "precipitacion_3d_mm",
    "precipitacion_7d_mm",
    "humedad_suelo_pct",
    "cobertura_vegetal_pct",
    "superficie_impermeable_pct",
    "densidad_construccion_pct",
    "tipo_uso_suelo",
    "estado_drenaje",
    "historial_inundacion_eventos",
]

TARGET = "riesgo"
CATEGORICAL = ["tipo_uso_suelo", "estado_drenaje"]
NUMERIC = [col for col in FEATURES if col not in CATEGORICAL]
COORDINATES = ["latitud", "longitud"]

USO_SUELO = [
    "Residencial",
    "Comercial",
    "Mixto",
    "Institucional",
    "Recreativo",
    "Industrial",
]
ESTADO_DRENAJE = ["Bueno", "Regular", "Deficiente", "Colapsado"]

LABELS = {
    "distancia_canada_m": "Distancia a la Q. La bermejala",
    "distancia_drenaje_m": "Distancia al drenaje",
    "elevacion_msnm": "Elevación",
    "diferencia_elevacion_m": "Diferencia de elevación",
    "pendiente_pct": "Pendiente",
    "precipitacion_1h_mm": "Precipitación 1 h",
    "precipitacion_6h_mm": "Precipitación 6 h",
    "precipitacion_24h_mm": "Precipitación 24 h",
    "precipitacion_3d_mm": "Precipitación 3 días",
    "precipitacion_7d_mm": "Precipitación 7 días",
    "humedad_suelo_pct": "Humedad del suelo",
    "cobertura_vegetal_pct": "Cobertura vegetal",
    "superficie_impermeable_pct": "Superficie impermeable",
    "densidad_construccion_pct": "Densidad de construcción",
    "tipo_uso_suelo": "Uso del suelo",
    "estado_drenaje": "Estado del drenaje",
    "historial_inundacion_eventos": "Historial de inundación",
    "riesgo": "Riesgo",
    "latitud": "Latitud",
    "longitud": "Longitud",
}

UNITS = {
    "distancia_canada_m": "m",
    "distancia_drenaje_m": "m",
    "elevacion_msnm": "msnm",
    "diferencia_elevacion_m": "m",
    "pendiente_pct": "%",
    "precipitacion_1h_mm": "mm",
    "precipitacion_6h_mm": "mm",
    "precipitacion_24h_mm": "mm",
    "precipitacion_3d_mm": "mm",
    "precipitacion_7d_mm": "mm",
    "humedad_suelo_pct": "%",
    "cobertura_vegetal_pct": "%",
    "superficie_impermeable_pct": "%",
    "densidad_construccion_pct": "%",
    "historial_inundacion_eventos": "eventos",
}

PRECIP_CHAIN = [
    "precipitacion_1h_mm",
    "precipitacion_6h_mm",
    "precipitacion_24h_mm",
    "precipitacion_3d_mm",
    "precipitacion_7d_mm",
]


def label(column: str) -> str:
    return LABELS.get(column, column.replace("_", " "))
