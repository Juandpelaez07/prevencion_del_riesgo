"""Formato numérico en español."""

from src.schema import CATEGORICAL, UNITS


def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def fmt_num(value: float, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{fmt_num(value * 100, digits)} %"


def fmt_measure(column: str, value) -> str:
    if column in CATEGORICAL:
        return str(value)
    number = float(value)
    if column == "historial_inundacion_eventos":
        n = int(round(number))
        return f"{n} evento" if n == 1 else f"{n} eventos"
    digits = 0 if abs(number) >= 10 else 1
    unit = UNITS.get(column, "")
    text = fmt_num(number, digits)
    return f"{text} {unit}".strip()
