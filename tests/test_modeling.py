import pandas as pd

from src.dataset import build_dataset
from src.modeling import MODEL_ORDER, train_all
from src.recommendations import row_from_mapping
from src.schema import FEATURES


def _fila(**overrides):
    base = {column: 0 for column in FEATURES}
    base.update(
        {
            "distancia_canada_m": 180,
            "distancia_drenaje_m": 40,
            "elevacion_msnm": 1560,
            "diferencia_elevacion_m": 12,
            "pendiente_pct": 8,
            "tipo_uso_suelo": "Recreativo",
            "estado_drenaje": "Bueno",
            "cobertura_vegetal_pct": 40,
            "superficie_impermeable_pct": 30,
            "densidad_construccion_pct": 25,
        }
    )
    base.update(overrides)
    return row_from_mapping(base)


def test_los_tres_modelos_entrenan_y_separan_un_aguacero_de_un_dia_seco():
    frame = build_dataset(n=1400, seed=11)
    bundle = train_all(frame, fast=True)
    assert list(bundle.reports) == MODEL_ORDER
    for name, report in bundle.reports.items():
        for stats in (report.metrics_default, report.metrics_tuned):
            assert 0 <= stats["recall"] <= 1
            assert stats["roc_auc"] > 0.75
            assert stats["matriz"].shape == (2, 2)
        assert report.importances["variable"].is_unique or len(report.importances) > 5

    seco = _fila()
    aguacero = _fila(
        distancia_canada_m=16,
        distancia_drenaje_m=8,
        diferencia_elevacion_m=0.8,
        elevacion_msnm=1505,
        pendiente_pct=18,
        precipitacion_1h_mm=42,
        precipitacion_6h_mm=58,
        precipitacion_24h_mm=76,
        precipitacion_3d_mm=94,
        precipitacion_7d_mm=120,
        humedad_suelo_pct=84,
        cobertura_vegetal_pct=8,
        superficie_impermeable_pct=86,
        densidad_construccion_pct=80,
        tipo_uso_suelo="Residencial",
        estado_drenaje="Colapsado",
        historial_inundacion_eventos=4,
    )
    model = bundle.recommended
    p_seco = float(bundle.reports[model].pipeline.predict_proba(seco)[0, 1])
    p_agua = float(bundle.reports[model].pipeline.predict_proba(aguacero)[0, 1])
    assert p_agua > p_seco + 0.2
    assert isinstance(bundle.reference, pd.DataFrame)
