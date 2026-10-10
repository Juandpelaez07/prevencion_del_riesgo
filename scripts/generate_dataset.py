"""Genera el CSV de demostración y el plano hídrico de referencia."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import DEFAULT_CSV, build_dataset
from src.geography import SECTORS, in_comuna, sector_profile
from src.mapping import save_reference_map


def main() -> None:
    frame = build_dataset()
    DEFAULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DEFAULT_CSV, index=False, encoding="utf-8-sig")
    image = save_reference_map()
    rate = frame["riesgo"].mean()
    print(f"filas={len(frame)} riesgo={rate:.4f} csv={DEFAULT_CSV}")
    print(f"mapa={image}")
    for sector in SECTORS:
        profile = sector_profile(sector)
        inside = in_comuna(profile["latitud"], profile["longitud"])
        print(
            f"{sector.nombre}: lat={profile['latitud']:.5f} lon={profile['longitud']:.5f} "
            f"dist={profile['distancia_canada_m']} elev={profile['elevacion_msnm']} dentro={inside}"
        )


if __name__ == "__main__":
    main()
