"""Geometría de referencia de la Q. La bermejala de estudio en Aranjuez."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from matplotlib.path import Path

LAT0 = 6.282
LON0 = -75.560
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * float(np.cos(np.deg2rad(LAT0)))

# De la ladera oriental (aguas arriba) hacia el río Medellín (aguas abajo).
CHANNEL = np.array(
    [
        [6.2934, -75.5476],
        [6.2908, -75.5498],
        [6.2876, -75.5528],
        [6.2846, -75.5562],
        [6.2818, -75.5596],
        [6.2790, -75.5630],
        [6.2764, -75.5664],
        [6.2740, -75.5696],
        [6.2716, -75.5726],
    ],
    dtype=float,
)

COMUNA = np.array(
    [
        [6.2678, -75.5748],
        [6.2674, -75.5638],
        [6.2692, -75.5535],
        [6.2736, -75.5462],
        [6.2810, -75.5444],
        [6.2898, -75.5454],
        [6.2968, -75.5496],
        [6.3002, -75.5572],
        [6.2988, -75.5678],
        [6.2920, -75.5742],
        [6.2824, -75.5758],
        [6.2738, -75.5756],
    ],
    dtype=float,
)

UPSTREAM_ELEV = 1638.0
DOWNSTREAM_ELEV = 1484.0


@dataclass(frozen=True)
class Sector:
    nombre: str
    along_frac: float
    offset_m: float
    banco_m: float
    uso: str
    drenaje: str
    vegetal: float
    impermeable: float
    construccion: float
    historial: int
    nota: str


SECTORS: tuple[Sector, ...] = (
    Sector(
        "Las Esmeraldas",
        0.08,
        150,
        5.5,
        "Residencial",
        "Regular",
        38,
        52,
        48,
        0,
        "Ladera alta, con más jardín y la Q. La bermejala todavía encajada.",
    ),
    Sector(
        "Berlín",
        0.18,
        62,
        3.2,
        "Residencial",
        "Regular",
        24,
        64,
        58,
        1,
        "Residencial de ladera. El retiro se estrecha en algunas cuadras.",
    ),
    Sector(
        "Brasilia",
        0.30,
        96,
        3.8,
        "Residencial",
        "Bueno",
        28,
        60,
        55,
        0,
        "Tejido residencial compacto, a una distancia media del cauce.",
    ),
    Sector(
        "San Isidro",
        0.40,
        34,
        1.4,
        "Residencial",
        "Deficiente",
        16,
        74,
        70,
        2,
        "Casas cercanas a la margen, con pendiente todavía marcada.",
    ),
    Sector(
        "Palermo",
        0.50,
        78,
        2.6,
        "Mixto",
        "Regular",
        18,
        72,
        66,
        1,
        "Uso mixto, con comercio sobre las vías que cruzan hacia el cauce.",
    ),
    Sector(
        "Bermejal",
        0.60,
        16,
        0.6,
        "Residencial",
        "Deficiente",
        8,
        82,
        76,
        3,
        "Borde inmediato de la Q. La bermejala. Es el supuesto más expuesto del recorrido.",
    ),
    Sector(
        "Miranda",
        0.70,
        44,
        1.6,
        "Residencial",
        "Deficiente",
        12,
        80,
        78,
        2,
        "Densidad alta y poca vegetación entre la calle y el cauce.",
    ),
    Sector(
        "San Pedro",
        0.78,
        28,
        1.0,
        "Mixto",
        "Colapsado",
        10,
        84,
        80,
        4,
        "Parte baja. El supuesto de trabajo incluye drenaje sin capacidad.",
    ),
    Sector(
        "Sevilla",
        0.86,
        120,
        3.4,
        "Comercial",
        "Regular",
        14,
        76,
        68,
        1,
        "Un poco más retirado del eje, con actividad comercial.",
    ),
    Sector(
        "Moravia",
        0.93,
        52,
        1.1,
        "Residencial",
        "Deficiente",
        9,
        88,
        86,
        3,
        "Zona baja y muy construida, donde el agua ya llega con velocidad.",
    ),
)

SECTOR_BY_NAME = {sector.nombre: sector for sector in SECTORS}


def project(lat, lon):
    lat_a = np.asarray(lat, dtype=float)
    lon_a = np.asarray(lon, dtype=float)
    x = (lon_a - LON0) * M_PER_DEG_LON
    y = (lat_a - LAT0) * M_PER_DEG_LAT
    return x, y


def unproject(x, y):
    x_a = np.asarray(x, dtype=float)
    y_a = np.asarray(y, dtype=float)
    lon = x_a / M_PER_DEG_LON + LON0
    lat = y_a / M_PER_DEG_LAT + LAT0
    return lat, lon


@lru_cache(maxsize=1)
def _frames():
    x, y = project(CHANNEL[:, 0], CHANNEL[:, 1])
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return x, y, seg, cum


def channel_length() -> float:
    return float(_frames()[3][-1])


def thalweg_elevation(along_m) -> np.ndarray:
    length = channel_length()
    frac = np.clip(np.asarray(along_m, dtype=float) / length, 0, 1)
    return UPSTREAM_ELEV + (DOWNSTREAM_ELEV - UPSTREAM_ELEV) * frac


def point_at(along_m, offset_m):
    scalar = np.isscalar(along_m) and np.isscalar(offset_m)
    along = np.atleast_1d(np.asarray(along_m, dtype=float))
    offset = np.broadcast_to(np.asarray(offset_m, dtype=float), along.shape).copy()
    x, y, seg, cum = _frames()
    along = np.clip(along, 0, cum[-1] - 1e-6)
    idx = np.clip(np.searchsorted(cum, along, side="right") - 1, 0, len(seg) - 1)
    step = seg[idx]
    t = (along - cum[idx]) / step
    px = x[idx] + t * (x[idx + 1] - x[idx])
    py = y[idx] + t * (y[idx + 1] - y[idx])
    tx = (x[idx + 1] - x[idx]) / step
    ty = (y[idx + 1] - y[idx]) / step
    qx = px + offset * (-ty)
    qy = py + offset * tx
    lat, lon = unproject(qx, qy)
    if scalar:
        return float(lat[0]), float(lon[0])
    return lat, lon


def locate_many(lat, lon):
    """Distancia en metros al eje y abscisa a lo largo de la Q. La bermejala."""
    x, y = project(lat, lon)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    sx, sy, seg, cum = _frames()
    best_d2 = np.full(x.shape, np.inf)
    best_s = np.zeros(x.shape)
    for i, step in enumerate(seg):
        vx = sx[i + 1] - sx[i]
        vy = sy[i + 1] - sy[i]
        t = np.clip(((x - sx[i]) * vx + (y - sy[i]) * vy) / (step**2), 0, 1)
        px = sx[i] + t * vx
        py = sy[i] + t * vy
        d2 = (x - px) ** 2 + (y - py) ** 2
        better = d2 < best_d2
        best_d2 = np.where(better, d2, best_d2)
        best_s = np.where(better, cum[i] + t * step, best_s)
    return np.sqrt(best_d2), best_s


def comuna_path() -> Path:
    return Path(np.column_stack([COMUNA[:, 1], COMUNA[:, 0]]))


def in_comuna(lat, lon) -> np.ndarray:
    pts = np.column_stack(
        [np.atleast_1d(lon).astype(float), np.atleast_1d(lat).astype(float)]
    )
    inside = comuna_path().contains_points(pts)
    if np.isscalar(lat) and np.isscalar(lon):
        return bool(inside[0])
    return inside


def buffer_ring(distance_m: float, samples: int = 90) -> np.ndarray:
    length = channel_length()
    along = np.linspace(0, length * 0.999, samples)
    lat_l, lon_l = point_at(along, distance_m)
    lat_r, lon_r = point_at(along, -distance_m)
    lats = np.concatenate([lat_l, lat_r[::-1]])
    lons = np.concatenate([lon_l, lon_r[::-1]])
    return np.column_stack([lats, lons])


def sector_location(sector: Sector) -> tuple[float, float, float, float]:
    along = sector.along_frac * channel_length()
    lat, lon = point_at(along, sector.offset_m)
    return lat, lon, abs(sector.offset_m), float(thalweg_elevation(along))


def sector_profile(sector: Sector) -> dict:
    lat, lon, dist, thalweg = sector_location(sector)
    diferencia = max(0.6, 0.038 * dist + sector.banco_m)
    elevacion = thalweg + diferencia
    pendiente = float(
        np.clip(7 + 20 * (1 - sector.along_frac) + 0.012 * dist, 2, 46)
    )
    return {
        "sector": sector.nombre,
        "nota": sector.nota,
        "latitud": lat,
        "longitud": lon,
        "distancia_canada_m": round(dist, 1),
        "distancia_drenaje_m": round(max(4.0, dist * 0.42), 1),
        "elevacion_msnm": round(elevacion, 1),
        "diferencia_elevacion_m": round(diferencia, 1),
        "pendiente_pct": round(pendiente, 1),
        "tipo_uso_suelo": sector.uso,
        "estado_drenaje": sector.drenaje,
        "cobertura_vegetal_pct": sector.vegetal,
        "superficie_impermeable_pct": sector.impermeable,
        "densidad_construccion_pct": sector.construccion,
        "historial_inundacion_eventos": sector.historial,
    }


def reconstruct_coordinates(distancia_m, seed: int = 7):
    """Ubica puntos a la distancia indicada para poder dibujarlos si el CSV no trae coordenadas."""
    dist = np.asarray(distancia_m, dtype=float)
    rng = np.random.default_rng(seed)
    along = rng.uniform(0, channel_length() * 0.999, size=len(dist))
    side = rng.choice(np.array([-1.0, 1.0]), size=len(dist))
    return point_at(along, side * dist)
