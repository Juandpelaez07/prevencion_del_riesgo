import pandas as pd

from src.recommendations import LEVELS, classify, contextual_notes
from src.schema import FEATURES


def test_las_franjas_cubren_la_probabilidad():
    assert classify(0.0).key == "bajo"
    assert classify(0.249).key == "bajo"
    assert classify(0.25).key == "medio"
    assert classify(0.5).key == "alto"
    assert classify(0.749).key == "alto"
    assert classify(0.75).key == "muy_alto"
    assert classify(1.0).key == "muy_alto"
    assert [level.title for level in LEVELS] == [
        "Monitoreo normal",
        "Seguimiento de lluvias",
        "Evitar actividades cercanas al cauce",
        "Priorizar medidas preventivas y atención a alertas oficiales",
    ]


def test_una_margen_con_aguacero_agrega_consejos_concretos():
    row = pd.Series({column: 0.0 for column in FEATURES}, dtype=object)
    row["distancia_canada_m"] = 12
    row["diferencia_elevacion_m"] = 1.2
    row["precipitacion_1h_mm"] = 35
    row["estado_drenaje"] = "Colapsado"
    row["historial_inundacion_eventos"] = 3
    row["humedad_suelo_pct"] = 88
    notes = contextual_notes(row)
    assert len(notes) >= 4
    assert any("minutos" in note for note in notes)
    assert any("alcantarilla" in note for note in notes)
