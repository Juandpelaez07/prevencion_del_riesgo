import numpy as np

from src.geography import SECTORS, channel_length, in_comuna, locate_many, point_at, sector_profile


def test_un_punto_sobre_el_eje_esta_a_cero_metros():
    lat, lon = point_at(120.0, 0.0)
    dist, _along = locate_many(lat, lon)
    assert dist[0] < 1.0


def test_el_desplazamiento_perpendicular_conserva_la_distancia():
    lat, lon = point_at(400.0, -75.0)
    dist, _along = locate_many(lat, lon)
    assert abs(dist[0] - 75.0) < 1.5


def test_la_canada_tiene_unos_pocos_kilometros():
    assert 2500 < channel_length() < 6000


def test_los_sectores_caen_dentro_de_la_comuna():
    for sector in SECTORS:
        profile = sector_profile(sector)
        assert in_comuna(profile["latitud"], profile["longitud"]), sector.nombre
        assert profile["distancia_canada_m"] == abs(sector.offset_m)


def test_point_at_vectorizado():
    lat, lon = point_at(np.array([50.0, 800.0]), np.array([10.0, -20.0]))
    dist, _along = locate_many(lat, lon)
    assert abs(dist[0] - 10) < 1.5
    assert abs(dist[1] - 20) < 1.5
