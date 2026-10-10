"""Dataset sintético y lectura de un CSV con el mismo esquema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.geography import (
    channel_length,
    in_comuna,
    point_at,
    reconstruct_coordinates,
    thalweg_elevation,
)
from src.schema import (
    CATEGORICAL,
    COORDINATES,
    ESTADO_DRENAJE,
    FEATURES,
    NUMERIC,
    TARGET,
    USO_SUELO,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "Prevencion_del_riesgo.csv"
ACCENT_CSV = ROOT / "data" / "Prevención_del_riesgo.csv"
TARGET_RATE = 0.288
LABEL_NOISE = 0.035

USO_CANON = {name.lower(): name for name in USO_SUELO}
DRENAJE_CANON = {name.lower(): name for name in ESTADO_DRENAJE}
DRENAJE_CANON.update({"deficiente": "Deficiente", "colapsado": "Colapsado"})


class DatasetError(ValueError):
    pass


def _sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def _risk_logit(frame: pd.DataFrame, intercept: float) -> np.ndarray:
    uso = frame["tipo_uso_suelo"].map(
        {
            "Recreativo": -0.35,
            "Institucional": -0.12,
            "Industrial": 0.05,
            "Residencial": 0.18,
            "Mixto": 0.22,
            "Comercial": 0.16,
        }
    ).astype(float)
    drenaje = frame["estado_drenaje"].map(
        {
            "Bueno": -0.55,
            "Regular": -0.05,
            "Deficiente": 0.48,
            "Colapsado": 0.95,
        }
    ).astype(float)
    return (
        intercept
        - 1.55 * np.log1p(frame["distancia_canada_m"]) / np.log1p(250)
        - 0.45 * np.log1p(frame["distancia_drenaje_m"]) / np.log1p(120)
        - 1.25 * np.clip(frame["diferencia_elevacion_m"] / 10, 0, 3)
        + 0.28 * np.clip(frame["pendiente_pct"] / 25, 0, 2)
        + 1.35 * np.tanh(frame["precipitacion_1h_mm"] / 16)
        + 0.55 * np.tanh(frame["precipitacion_6h_mm"] / 30)
        + 0.75 * np.tanh(frame["precipitacion_24h_mm"] / 40)
        + 0.25 * np.tanh(frame["precipitacion_3d_mm"] / 70)
        + 0.40 * np.tanh(frame["precipitacion_7d_mm"] / 90)
        + 0.85 * (frame["humedad_suelo_pct"] / 100)
        - 0.55 * (frame["cobertura_vegetal_pct"] / 100)
        + 0.65 * (frame["superficie_impermeable_pct"] / 100)
        + 0.30 * (frame["densidad_construccion_pct"] / 100)
        + 0.22 * np.minimum(frame["historial_inundacion_eventos"], 8)
        + uso.to_numpy()
        + drenaje.to_numpy()
    )


def _sample_categories(rng: np.random.Generator, urban: np.ndarray, dist: np.ndarray):
    n = len(urban)
    uso = []
    drenaje = []
    for i in range(n):
        u = float(urban[i])
        pesos = np.array(
            [
                0.42 + 0.22 * u,  # Residencial
                0.10 + 0.08 * u,  # Comercial
                0.14,  # Mixto
                0.12 * (1 - 0.35 * u),  # Institucional
                0.16 * (1 - 0.45 * u),  # Recreativo
                0.07,  # Industrial
            ]
        )
        uso.append(USO_SUELO[int(rng.choice(len(USO_SUELO), p=pesos / pesos.sum()))])
        cercania = np.clip((80 - dist[i]) / 80, 0, 1)
        malo = 0.30 * u + 0.35 * cercania + float(rng.random())
        if malo < 0.45:
            drenaje.append("Bueno")
        elif malo < 0.85:
            drenaje.append("Regular")
        elif malo < 1.25:
            drenaje.append("Deficiente")
        else:
            drenaje.append("Colapsado")
    return np.array(uso), np.array(drenaje)


def build_dataset(n: int = 4200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    length = channel_length()
    along = rng.uniform(0, length * 0.998, n)
    roll = rng.random(n)
    dist = np.empty(n)
    near = roll < 0.40
    mid = (roll >= 0.40) & (roll < 0.80)
    far = roll >= 0.80
    dist[near] = rng.exponential(36, int(near.sum())) + 4
    dist[mid] = rng.uniform(30, 210, int(mid.sum()))
    dist[far] = rng.uniform(210, 760, int(far.sum()))
    dist = np.clip(dist, 3, 820)
    side = rng.choice(np.array([-1.0, 1.0]), size=n)
    lat, lon = point_at(along, side * dist)

    # Si el punto cae fuera de la comuna, se refleja hacia el eje.
    outside = ~in_comuna(lat, lon)
    if outside.any():
        lat_in, lon_in = point_at(along[outside], side[outside] * np.minimum(dist[outside], 180))
        lat = lat.copy()
        lon = lon.copy()
        lat[outside] = lat_in
        lon[outside] = lon_in
        dist = dist.copy()
        dist[outside] = np.minimum(dist[outside], 180)

    thalweg = thalweg_elevation(along)
    diferencia = np.clip(0.034 * dist + rng.normal(4.2, 2.4, n), 0.3, 42)
    elevacion = np.clip(thalweg + diferencia, 1476, 1685)
    pendiente = np.clip(
        5.5 + 18 * (1 - along / length) + 0.012 * dist + rng.normal(0, 3.2, n),
        1.0,
        52,
    )
    drenaje_dist = np.clip(dist * rng.uniform(0.18, 0.75, n) + rng.normal(0, 6, n), 2, 380)

    urban = np.clip(0.78 - (elevacion - 1484) / 520 + rng.normal(0, 0.08, n), 0.12, 0.97)
    impermeable = np.clip(urban * 92 + rng.normal(0, 7, n), 8, 98)
    vegetal = np.clip(76 - 0.62 * impermeable + rng.normal(0, 6, n), 0, 88)
    construccion = np.clip(urban * 88 + rng.normal(0, 6, n), 6, 98)
    uso, drenaje = _sample_categories(rng, urban, dist)

    drenaje_peso = pd.Series(drenaje).map(
        {"Bueno": 0.15, "Regular": 0.35, "Deficiente": 0.7, "Colapsado": 1.0}
    ).to_numpy()
    lam = (
        0.25
        + 2.4 * np.exp(-dist / 55)
        + 0.9 * (diferencia < 4)
        + 0.45 * drenaje_peso
    )
    historial = np.clip(rng.poisson(lam), 0, 12)

    storm = rng.gamma(1.15, 1.0, n)
    p1 = rng.gamma(1.15, 2.35, n) * (0.32 + storm)
    extremos = rng.random(n) < 0.028
    p1 = np.where(extremos, p1 + rng.uniform(22, 68, n), p1)
    p1 = np.clip(p1, 0, 115)
    p6 = p1 + rng.gamma(1.25, 2.15, n) * (0.45 + 0.35 * storm)
    p24 = p6 + rng.gamma(1.35, 3.1, n) * (0.55 + 0.45 * storm)
    p3 = p24 + rng.gamma(1.4, 4.2, n)
    p7 = p3 + rng.gamma(1.5, 5.8, n)
    humedad = np.clip(22 + 0.38 * p7 + rng.normal(0, 6, n), 8, 98)

    frame = pd.DataFrame(
        {
            "distancia_canada_m": dist,
            "distancia_drenaje_m": drenaje_dist,
            "elevacion_msnm": elevacion,
            "diferencia_elevacion_m": diferencia,
            "pendiente_pct": pendiente,
            "precipitacion_1h_mm": p1,
            "precipitacion_6h_mm": p6,
            "precipitacion_24h_mm": p24,
            "precipitacion_3d_mm": p3,
            "precipitacion_7d_mm": p7,
            "humedad_suelo_pct": humedad,
            "cobertura_vegetal_pct": vegetal,
            "superficie_impermeable_pct": impermeable,
            "densidad_construccion_pct": construccion,
            "tipo_uso_suelo": uso,
            "estado_drenaje": drenaje,
            "historial_inundacion_eventos": historial,
        }
    )

    def expected_rate(intercept: float) -> float:
        prob = _sigmoid(_risk_logit(frame, intercept))
        return float(prob.mean() + LABEL_NOISE * (1 - 2 * prob.mean()))

    lo, hi = -3.5, 3.5
    for _ in range(48):
        mid = (lo + hi) / 2
        if expected_rate(mid) > TARGET_RATE:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2
    prob = _sigmoid(_risk_logit(frame, intercept))
    riesgo = rng.random(n) < prob
    flip = rng.random(n) < LABEL_NOISE
    riesgo = np.where(flip, ~riesgo, riesgo).astype(int)

    target_n = int(round(TARGET_RATE * n))
    diff = target_n - int(riesgo.sum())
    if diff != 0:
        order = np.argsort(np.abs(prob - 0.5))
        moved = 0
        want = 1 if diff > 0 else 0
        for idx in order:
            if riesgo[idx] != want:
                riesgo[idx] = want
                moved += 1
                if moved == abs(diff):
                    break

    frame[TARGET] = riesgo
    frame["latitud"] = np.round(lat, 6)
    frame["longitud"] = np.round(lon, 6)

    for col in NUMERIC:
        if col == "historial_inundacion_eventos":
            frame[col] = frame[col].round().astype(int)
        else:
            frame[col] = frame[col].round(1)

    order = rng.permutation(n)
    columns = FEATURES + [TARGET] + COORDINATES
    return frame.loc[order, columns].reset_index(drop=True)


def _parse_numbers(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.replace(" ", "", regex=False)
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    both = text.str.contains(r"\.", na=False) & text.str.contains(",", na=False)
    text = text.where(~both, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    comma = text.str.contains(",", na=False) & ~text.str.contains(r"\.", na=False)
    text = text.where(~comma, text.str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def _normalize_category(series: pd.Series, canon: dict[str, str]) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    mapped = cleaned.str.lower().map(canon)
    return mapped.fillna(cleaned)


def _map_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.dropna().isin([0, 1]).all():
            return values
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "0": 0,
            "0.0": 0,
            "1": 1,
            "1.0": 1,
            "no": 0,
            "sí": 1,
            "si": 1,
            "no riesgo": 0,
            "sin riesgo": 0,
            "riesgo": 1,
            "false": 0,
            "true": 1,
        }
    )
    return mapped


def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = df.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    missing = [col for col in FEATURES + [TARGET] if col not in frame.columns]
    if missing:
        names = ", ".join(missing)
        raise DatasetError(f"Faltan columnas: {names}.")

    for col in NUMERIC:
        frame[col] = _parse_numbers(frame[col])
    frame["tipo_uso_suelo"] = _normalize_category(frame["tipo_uso_suelo"], USO_CANON)
    frame["estado_drenaje"] = _normalize_category(frame["estado_drenaje"], DRENAJE_CANON)
    frame[TARGET] = _map_target(frame[TARGET])

    before = len(frame)
    frame = frame.dropna(subset=FEATURES + [TARGET]).copy()
    dropped = before - len(frame)
    if frame.empty:
        raise DatasetError("El archivo no dejó registros válidos después de revisar las columnas.")
    if not frame[TARGET].isin([0, 1]).all():
        raise DatasetError("La columna riesgo debe ser 0 (sin riesgo) o 1 (riesgo).")
    frame[TARGET] = frame[TARGET].astype(int)
    frame["historial_inundacion_eventos"] = frame["historial_inundacion_eventos"].round().astype(int)

    measured = False
    if set(COORDINATES).issubset(frame.columns):
        frame["latitud"] = _parse_numbers(frame["latitud"])
        frame["longitud"] = _parse_numbers(frame["longitud"])
        measured = bool(frame["latitud"].notna().all() and frame["longitud"].notna().all())
    if not measured:
        lat, lon = reconstruct_coordinates(frame["distancia_canada_m"].to_numpy())
        frame["latitud"] = np.round(lat, 6)
        frame["longitud"] = np.round(lon, 6)

    meta = {
        "filas_descartadas": int(dropped),
        "coordenadas": "medidas" if measured else "ilustrativas",
        "filas": int(len(frame)),
        "tasa_riesgo": float(frame[TARGET].mean()),
    }
    keep = FEATURES + [TARGET] + COORDINATES
    return frame[keep].reset_index(drop=True), meta


def read_table(source) -> pd.DataFrame:
    try:
        return pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(source, sep=None, engine="python", encoding="latin-1")
    except Exception as exc:  # noqa: BLE001 - el mensaje se muestra en la interfaz
        raise DatasetError(f"No se pudo leer el CSV: {exc}") from exc


def load_table(source) -> tuple[pd.DataFrame, dict]:
    return prepare_frame(read_table(source))


def default_source() -> Path:
    if DEFAULT_CSV.exists():
        return DEFAULT_CSV
    if ACCENT_CSV.exists():
        return ACCENT_CSV
    raise DatasetError(
        "No está el dataset de demostración. Generelo con python scripts/generate_dataset.py."
    )
