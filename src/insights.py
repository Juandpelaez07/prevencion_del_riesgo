"""Resúmenes del análisis exploratorio, calculados sobre el dataset activo."""

from __future__ import annotations

import pandas as pd

from src.format import fmt_pct
from src.schema import NUMERIC, TARGET, label


def correlation_with_target(df: pd.DataFrame) -> pd.Series:
    corr = df[NUMERIC + [TARGET]].corr(method="pearson")[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (df[col] < low) | (df[col] > high)
        rows.append(
            {
                "Variable": label(col),
                "Q1": round(q1, 1),
                "Q3": round(q3, 1),
                "Límite inferior": round(low, 1),
                "Límite superior": round(high, 1),
                "Atípicos": int(mask.sum()),
                "%": round(100 * mask.mean(), 1),
            }
        )
    table = pd.DataFrame(rows).sort_values(["Atípicos", "Variable"], ascending=[False, True])
    return table.reset_index(drop=True)


def binned_risk(df: pd.DataFrame, column: str, edges, names) -> pd.DataFrame:
    groups = pd.cut(df[column], bins=edges, labels=names, include_lowest=True)
    summary = (
        df.groupby(groups, observed=False)[TARGET]
        .agg(puntos="size", en_riesgo="sum", tasa="mean")
        .reset_index()
    )
    summary = summary.rename(columns={column: "tramo", "tasa": "proporcion"})
    summary["tramo"] = summary["tramo"].astype(str)
    summary["proporcion"] = summary["proporcion"].fillna(0)
    return summary


def rate_where(df: pd.DataFrame, mask: pd.Series) -> float | None:
    if int(mask.sum()) < 25:
        return None
    return float(df.loc[mask, TARGET].mean())


def highlights(df: pd.DataFrame) -> list[str]:
    lines = []
    corr = correlation_with_target(df)
    top = corr.index[0]
    lines.append(
        f"La variable numérica más asociada con el riesgo es {label(top).lower()} "
        f"(correlación de {corr.iloc[0]:+.2f})."
    )
    near = rate_where(df, df["distancia_canada_m"] <= 30)
    far = rate_where(df, df["distancia_canada_m"] >= 200)
    if near is not None and far is not None:
        lines.append(
            f"A 30 m o menos de la Q. La bermejala, {fmt_pct(near)} de los puntos queda en riesgo. "
            f"A 200 m o más, la proporción es {fmt_pct(far)}."
        )
    low = rate_where(df, df["diferencia_elevacion_m"] <= 3)
    high = rate_where(df, df["diferencia_elevacion_m"] >= 12)
    if low is not None and high is not None:
        lines.append(
            f"Cuando el punto está a 3 m o menos por encima del cauce, el riesgo aparece en {fmt_pct(low)} de los casos. "
            f"Con 12 m o más de diferencia, baja a {fmt_pct(high)}."
        )
    wet = rate_where(df, df["precipitacion_1h_mm"] >= 20)
    dry = rate_where(df, df["precipitacion_1h_mm"] < 2)
    if wet is not None and dry is not None:
        lines.append(
            f"Con 20 mm o más en una hora, {fmt_pct(wet)} de los puntos está en riesgo. "
            f"Con menos de 2 mm, esa cifra es {fmt_pct(dry)}."
        )
    rate = float(df[TARGET].mean())
    lines.append(
        f"El {fmt_pct(1 - rate)} de los registros no tiene riesgo y el {fmt_pct(rate)} sí. "
        f"Por eso la exactitud sola engaña: un modelo que siempre diga «sin riesgo» acertaría cerca del "
        f"{fmt_pct(1 - rate)} y no detectaría ningún punto expuesto."
    )
    pair = df[["superficie_impermeable_pct", "cobertura_vegetal_pct"]].corr().iloc[0, 1]
    if abs(pair) >= 0.7:
        lines.append(
            "La superficie impermeable y la cobertura vegetal se mueven casi al revés. "
            "Conviene leerlas juntas: donde se construye el retiro, desaparece la vegetación que frena el agua."
        )
    return lines


def category_risk(df: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        df.groupby(column, observed=False)[TARGET]
        .agg(puntos="size", proporcion="mean")
        .reset_index()
        .sort_values("proporcion", ascending=False)
    )
    return summary
