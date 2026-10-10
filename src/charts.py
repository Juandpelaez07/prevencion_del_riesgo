"""Gráficas con una misma paleta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from src.schema import CATEGORICAL, label

INK = "#17241c"
MUTED = "#5e6d64"
SAFE = "#2f6f4e"
RISK = "#c2412d"
WATER = "#1d6a8a"
AMBER = "#d97706"
GRID = "#e6dfd2"
CLASS_COLOR = {"Sin riesgo": SAFE, "Riesgo": RISK}


def _class_name(series: pd.Series) -> pd.Series:
    return series.map({0: "Sin riesgo", 1: "Riesgo"})


def apply_layout(fig, height: int = 420, title: str | None = None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,252,246,0.9)",
        font=dict(family="Outfit, Segoe UI, sans-serif", color=INK, size=13),
        title=dict(
            text=title or "",
            font=dict(family="Fraunces, Georgia, serif", size=20, color=INK),
            x=0,
            xanchor="left",
        ),
        margin=dict(l=12, r=12, t=56 if title else 24, b=12),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
        colorway=[SAFE, RISK, WATER, AMBER, "#3f4f46", "#8a6a3b"],
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(size=13))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(size=13))
    return fig


def risk_donut(df: pd.DataFrame):
    counts = df["riesgo"].value_counts().rename({0: "Sin riesgo", 1: "Riesgo"})
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.62,
            marker=dict(colors=[SAFE if name == "Sin riesgo" else RISK for name in counts.index]),
            sort=False,
            textinfo="percent",
            hovertemplate="%{label}<br>%{value} puntos<extra></extra>",
        )
    )
    fig.update_layout(showlegend=True)
    return apply_layout(fig, height=340, title="Distribución del riesgo")


def distribution(df: pd.DataFrame, column: str):
    frame = df.copy()
    frame["clase"] = _class_name(frame["riesgo"])
    title = label(column)
    if column in CATEGORICAL:
        counts = frame.groupby([column, "clase"], observed=False).size().reset_index(name="puntos")
        fig = px.bar(
            counts,
            x=column,
            y="puntos",
            color="clase",
            barmode="group",
            color_discrete_map=CLASS_COLOR,
        )
    else:
        fig = px.histogram(
            frame,
            x=column,
            color="clase",
            barmode="overlay",
            opacity=0.72,
            nbins=36,
            color_discrete_map=CLASS_COLOR,
        )
    fig.update_layout(xaxis_title=title, yaxis_title="Puntos", bargap=0.08)
    return apply_layout(fig, height=420, title=title)


def correlation_bars(corr: pd.Series):
    frame = corr.rename("correlacion").reset_index()
    frame.columns = ["variable", "correlacion"]
    frame["variable"] = frame["variable"].map(label)
    frame["sentido"] = np.where(frame["correlacion"] >= 0, "Sube con el riesgo", "Baja con el riesgo")
    fig = px.bar(
        frame,
        x="correlacion",
        y="variable",
        color="sentido",
        orientation="h",
        color_discrete_map={"Sube con el riesgo": RISK, "Baja con el riesgo": WATER},
    )
    fig.update_layout(xaxis_title="Correlación con riesgo", yaxis_title="", yaxis=dict(autorange="reversed"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    return apply_layout(fig, height=520, title="Qué variables se mueven con el riesgo")


def correlation_heatmap(df: pd.DataFrame):
    cols = [
        "distancia_canada_m",
        "diferencia_elevacion_m",
        "pendiente_pct",
        "precipitacion_1h_mm",
        "precipitacion_24h_mm",
        "precipitacion_7d_mm",
        "humedad_suelo_pct",
        "cobertura_vegetal_pct",
        "superficie_impermeable_pct",
        "densidad_construccion_pct",
        "historial_inundacion_eventos",
        "riesgo",
    ]
    matrix = df[cols].corr().rename(index=label, columns=label)
    fig = px.imshow(
        matrix,
        zmin=-1,
        zmax=1,
        color_continuous_scale=["#1d6a8a", "#f7f4ee", "#c2412d"],
        aspect="auto",
    )
    fig.update_layout(coloraxis_colorbar=dict(title=""))
    return apply_layout(fig, height=640, title="Correlaciones")


def box_by_class(df: pd.DataFrame, column: str):
    frame = df.copy()
    frame["clase"] = _class_name(frame["riesgo"])
    fig = px.box(
        frame,
        x="clase",
        y=column,
        color="clase",
        color_discrete_map=CLASS_COLOR,
        points=False,
    )
    fig.update_layout(xaxis_title="", yaxis_title=label(column), showlegend=False)
    return apply_layout(fig, height=420, title=f"Atípicos en {label(column).lower()}")


def binned_bars(summary: pd.DataFrame, title: str):
    fig = go.Figure(
        go.Bar(
            x=summary["tramo"],
            y=summary["proporcion"] * 100,
            marker_color=WATER,
            customdata=np.stack([summary["puntos"], summary["en_riesgo"]], axis=1),
            hovertemplate="%{x}<br>Riesgo: %{y:.1f}%<br>Puntos: %{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="", yaxis_title="% en riesgo", yaxis_range=[0, 100])
    return apply_layout(fig, height=380, title=title)


def category_bars(summary: pd.DataFrame, column: str):
    fig = px.bar(
        summary,
        x=column,
        y="proporcion",
        text=summary["puntos"],
        color_discrete_sequence=[WATER],
    )
    fig.update_traces(textposition="outside", texttemplate="%{text} pts")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Proporción en riesgo",
        yaxis_tickformat=".0%",
        yaxis_range=[0, min(1, float(summary["proporcion"].max()) + 0.18)],
    )
    return apply_layout(fig, height=380, title=label(column))


def scatter_risk(df: pd.DataFrame, x: str, y: str, title: str):
    frame = df.sample(n=min(len(df), 1800), random_state=7).copy()
    frame["clase"] = _class_name(frame["riesgo"])
    fig = px.scatter(
        frame,
        x=x,
        y=y,
        color="clase",
        opacity=0.55,
        color_discrete_map=CLASS_COLOR,
        hover_data={x: ":.1f", y: ":.1f"},
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(xaxis_title=label(x), yaxis_title=label(y))
    return apply_layout(fig, height=440, title=title)


def confusion_figure(matrix: np.ndarray, title: str):
    labels = ["Sin riesgo", "Riesgo"]
    text = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(f"{int(matrix[i, j])}")
        text.append(row)
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicho sin riesgo", "Predicho en riesgo"],
            y=["Real sin riesgo", "Real en riesgo"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#f7f4ee"], [1, "#1f6b4a"]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>%{z}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return apply_layout(fig, height=360, title=title)


def roc_figure(y_true, proba_by_model: dict[str, np.ndarray]):
    fig = go.Figure()
    colors = {
        "Regresión logística": AMBER,
        "Random Forest": WATER,
        "XGBoost": RISK,
    }
    for name, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, INK), width=2.4),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Azar",
            line=dict(color=MUTED, width=1, dash="dash"),
        )
    )
    fig.update_layout(xaxis_title="Falsos positivos", yaxis_title="Verdaderos positivos")
    return apply_layout(fig, height=440, title="Curvas ROC en el conjunto de prueba")


def importance_bars(frame: pd.DataFrame, signed: bool, title: str):
    top = frame.head(12).iloc[::-1]
    if signed:
        fig = go.Figure(
            go.Bar(
                x=top["valor"],
                y=top["variable"],
                orientation="h",
                marker_color=np.where(top["valor"] >= 0, RISK, WATER),
            )
        )
        fig.add_vline(x=0, line_color=INK, line_width=1)
        fig.update_layout(xaxis_title="Coeficiente (log-odds de riesgo)")
    else:
        fig = go.Figure(
            go.Bar(
                x=top["valor"],
                y=top["variable"],
                orientation="h",
                marker_color=WATER,
            )
        )
        fig.update_layout(xaxis_title="Importancia")
    fig.update_layout(yaxis_title="")
    return apply_layout(fig, height=460, title=title)


def response_figure(grid, values, column: str, model_name: str):
    fig = go.Figure(
        go.Scatter(
            x=grid,
            y=values * 100,
            mode="lines",
            line=dict(color=RISK, width=2.6),
            fill="tozeroy",
            fillcolor="rgba(194,65,45,0.12)",
        )
    )
    fig.update_layout(xaxis_title=label(column), yaxis_title="Probabilidad media de riesgo (%)")
    return apply_layout(fig, height=360, title=f"{model_name}: respuesta a {label(column).lower()}")
