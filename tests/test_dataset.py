import numpy as np

from src.dataset import build_dataset, prepare_frame
from src.geography import locate_many
from src.schema import FEATURES, PRECIP_CHAIN, TARGET


def test_esquema_balance_y_lluvia_acumulada():
    frame = build_dataset(n=900, seed=7)
    assert list(frame.columns[:18]) == FEATURES + [TARGET]
    assert abs(frame[TARGET].mean() - 0.288) < 0.002
    assert frame[FEATURES + [TARGET]].isna().sum().sum() == 0
    for left, right in zip(PRECIP_CHAIN, PRECIP_CHAIN[1:]):
        assert (frame[left] <= frame[right] + 1e-6).all()


def test_la_distancia_del_csv_coincide_con_el_plano():
    frame = build_dataset(n=400, seed=3)
    dist, _along = locate_many(frame["latitud"], frame["longitud"])
    assert np.max(np.abs(dist - frame["distancia_canada_m"].to_numpy())) < 1.5


def test_prepara_un_csv_con_decimales_de_coma_y_sin_coordenadas():
    raw = build_dataset(n=120, seed=1).drop(columns=["latitud", "longitud"])
    raw["distancia_canada_m"] = raw["distancia_canada_m"].map(lambda value: str(value).replace(".", ","))
    raw["riesgo"] = raw["riesgo"].map({0: "No riesgo", 1: "Riesgo"})
    frame, meta = prepare_frame(raw)
    assert meta["coordenadas"] == "ilustrativas"
    assert set(frame["riesgo"]).issubset({0, 1})
    assert frame["distancia_canada_m"].dtype.kind == "f"


def test_avisa_si_faltan_columnas():
    try:
        prepare_frame(__import__("pandas").DataFrame({"riesgo": [0, 1]}))
    except ValueError as exc:
        assert "Faltan columnas" in str(exc)
    else:
        raise AssertionError("debía fallar")
