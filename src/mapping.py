"""Mapa hídrico de referencia y mapa interactivo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.geography import (
    CHANNEL,
    COMUNA,
    LAT0,
    M_PER_DEG_LON,
    SECTORS,
    buffer_ring,
    channel_length,
    comuna_path,
    sector_profile,
)

ROOT = Path(__file__).resolve().parents[1]
# Plano real de la Q. La bermejala aportado por el usuario. No se genera ni se sobrescribe.
MAP_IMAGE = ROOT / "Images" / "mapa_hidrico.jpg"
# Salida del plano sintético que dibuja save_reference_map (solo referencia de trabajo).
GENERATED_MAP_IMAGE = ROOT / "Images" / "mapa_hidrico_generado.png"


def save_reference_map(path: Path = GENERATED_MAP_IMAGE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon, Rectangle

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 11.2), dpi=150)
    fig.patch.set_facecolor("#e7efe6")
    ax.set_facecolor("#d5e3d4")

    ax.add_patch(
        Polygon(np.column_stack([COMUNA[:, 1], COMUNA[:, 0]]), closed=True, facecolor="#f4f0e6", edgecolor="#1e3d32", linewidth=1.5, zorder=2)
    )

    river = np.array(
        [
            [-75.5802, 6.2668],
            [-75.5736, 6.2672],
            [-75.5724, 6.3012],
            [-75.5794, 6.3010],
        ]
    )
    ax.add_patch(Polygon(river, closed=True, facecolor="#9ec9d4", edgecolor="#5f93a3", linewidth=0.6, zorder=1))
    ax.text(-75.5772, 6.286, "Río\nMedellín", ha="center", va="center", fontsize=8, color="#1d4e5e", zorder=4)

    garden = Circle((-75.5662, 6.2706), 0.0023, facecolor="#b7d3ae", edgecolor="#5d8a62", linewidth=0.6, zorder=3)
    ax.add_patch(garden)
    ax.text(-75.5662, 6.2706, "Jardín\nBotánico", ha="center", va="center", fontsize=6.5, color="#234232", zorder=4)

    comuna = comuna_path()
    rng = np.random.default_rng(3)
    for lon in np.arange(-75.5748, -75.5452, 0.00255):
        for lat in np.arange(6.2690, 6.2990, 0.00205):
            if rng.random() < 0.22:
                continue
            jitter_x = float(rng.normal(0, 0.00045))
            jitter_y = float(rng.normal(0, 0.00028))
            if not comuna.contains_point((lon + jitter_x, lat + jitter_y)):
                continue
            w = float(rng.uniform(0.00075, 0.00145))
            h = float(rng.uniform(0.00042, 0.0009))
            tone = rng.choice(["#e7dccb", "#efe4d4", "#ddd1be"])
            block = Rectangle(
                (lon + jitter_x, lat + jitter_y),
                w,
                h,
                facecolor=tone,
                edgecolor="#e4d8c6",
                linewidth=0.15,
                zorder=3,
            )
            ax.add_patch(block)

    for distance, color, alpha in ((80, "#f0d7a4", 0.55), (30, "#efb5a2", 0.72)):
        ring = buffer_ring(distance)
        ax.add_patch(
            Polygon(
                np.column_stack([ring[:, 1], ring[:, 0]]),
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=4,
            )
        )

    ax.plot(CHANNEL[:, 1], CHANNEL[:, 0], color="#7ec3d4", linewidth=5.2, solid_capstyle="round", zorder=5)
    ax.plot(CHANNEL[:, 1], CHANNEL[:, 0], color="#165e78", linewidth=2.2, solid_capstyle="round", zorder=6)
    ax.annotate(
        "Q. La bermejala de estudio",
        xy=(CHANNEL[3, 1], CHANNEL[3, 0]),
        xytext=(CHANNEL[3, 1] + 0.0048, CHANNEL[3, 0] + 0.0042),
        fontsize=9,
        color="#165e78",
        arrowprops=dict(arrowstyle="-", color="#165e78", lw=0.7),
        zorder=8,
    )

    for sector in SECTORS:
        profile = sector_profile(sector)
        lon = profile["longitud"]
        lat = profile["latitud"]
        ax.scatter([lon], [lat], s=18, color="#17241c", zorder=7)
        dx = 0.0024 if sector.offset_m > 0 else -0.0024
        ha = "left" if sector.offset_m > 0 else "right"
        ax.text(lon + dx, lat + 0.0007, sector.nombre, fontsize=7.4, color="#17241c", ha=ha, va="bottom", zorder=8)

    ax.set_xlim(-75.5815, -75.5428)
    ax.set_ylim(6.2662, 6.3022)
    ax.set_aspect(1 / np.cos(np.deg2rad(LAT0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    scale_m = 400
    deg = scale_m / M_PER_DEG_LON
    x0, y0 = -75.5788, 6.2686
    ax.plot([x0, x0 + deg], [y0, y0], color="#17241c", lw=2.4, solid_capstyle="butt", zorder=9)
    ax.text(x0 + deg / 2, y0 + 0.0011, "400 m", ha="center", va="bottom", fontsize=8, color="#17241c", zorder=9)

    ax.annotate(
        "N",
        xy=(0.93, 0.16),
        xytext=(0.93, 0.09),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#17241c",
        arrowprops=dict(arrowstyle="-|>", color="#17241c", lw=1.2),
        zorder=9,
    )

    ax.text(
        0.025,
        0.975,
        "MAPA HÍDRICO DE REFERENCIA\nQ. La bermejala de estudio\nComuna 4 Aranjuez · Medellín",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        color="#17241c",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f7f4ee", edgecolor="#c9b89a", linewidth=0.8),
        zorder=10,
    )

    handles = [
        Line2D([0], [0], color="#165e78", lw=2.4, label="Q. La bermejala de estudio"),
        Line2D([0], [0], color="#efb5a2", lw=8, label="Franja de 30 m"),
        Line2D([0], [0], color="#f0d7a4", lw=8, label="Franja de 80 m"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#17241c", markersize=5, label="Sector de referencia"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#c9b89a",
        fontsize=8,
        borderpad=0.8,
    )
    legend.get_frame().set_facecolor("#f7f4ee")

    km = f"{channel_length() / 1000:.2f}".replace(".", ",")
    fig.text(
        0.06,
        0.025,
        "Plano de trabajo para el ejercicio. El trazo de la Q. La bermejala y los sectores son una referencia de análisis.\n"
        "No sustituye la cartografía oficial del DAP ni las alertas del DAGRD y de SIATA.  "
        f"Longitud del eje: {km} km.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#3e5148",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return path


def folium_map(df: pd.DataFrame, point: dict | None = None):
    import folium

    mapa = folium.Map(
        location=[6.283, -75.560],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.Polygon(
        locations=COMUNA.tolist(),
        color="#1f6b4a",
        weight=2,
        fill=False,
        tooltip="Comuna 4 Aranjuez",
    ).add_to(mapa)
    for distance, color in ((80, "#e0b15a"), (30, "#d4652f")):
        ring = buffer_ring(distance)
        folium.Polygon(
            locations=ring.tolist(),
            color=color,
            weight=1,
            fill=True,
            fill_opacity=0.18,
            tooltip=f"Franja de {distance} m",
        ).add_to(mapa)
    folium.PolyLine(
        locations=CHANNEL.tolist(),
        color="#165e78",
        weight=5,
        opacity=0.95,
        tooltip="Q. La bermejala de estudio",
    ).add_to(mapa)

    if point is None:
        sample = df.sample(n=min(650, len(df)), random_state=11)
        for _, row in sample.iterrows():
            risk = int(row["riesgo"])
            folium.CircleMarker(
                location=[row["latitud"], row["longitud"]],
                radius=4,
                color="#c2412d" if risk else "#2f6f4e",
                fill=True,
                fill_opacity=0.75,
                weight=0,
                tooltip="Riesgo" if risk else "Sin riesgo",
            ).add_to(mapa)
    else:
        folium.CircleMarker(
            location=[point["latitud"], point["longitud"]],
            radius=9,
            color="#17241c",
            fill=True,
            fill_color="#d4652f",
            fill_opacity=1,
            weight=2,
            tooltip=point.get("sector", "Punto consultado"),
        ).add_to(mapa)
        mapa.location = [point["latitud"], point["longitud"]]
        mapa.zoom_start = 15

    for sector in SECTORS:
        profile = sector_profile(sector)
        folium.Marker(
            location=[profile["latitud"], profile["longitud"]],
            tooltip=sector.nombre,
            icon=folium.DivIcon(
                html=(
                    "<div style='font-size:11px;color:#17241c;font-family:Georgia,serif;"
                    "white-space:nowrap;text-shadow:0 0 3px #fff'>"
                    f"{sector.nombre}</div>"
                )
            ),
        ).add_to(mapa)
    return mapa
