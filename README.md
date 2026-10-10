# Q. La bermejala Aranjuez — Prevención del riesgo

Aplicación para leer el riesgo de un punto frente a la Quebrada la bermejala de estudio en la Comuna 4, Aranjuez (Medellín). Cruza análisis exploratorio, tres modelos de clasificación y una recomendación según la probabilidad.

El dataset de demostración es sintético: cerca del 71,2 % de los puntos no tiene riesgo y el 28,8 % sí. Sirve para ver el desbalance y para no confundir la exactitud con la capacidad de detectar un punto expuesto.

## Cómo ejecutarlo en local

```bash
python -m venv .venv
.venv\Scripts\activate        # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run.py                 # equivale a: streamlit run app.py --server.port 8519
```

Abre [http://localhost:8519](http://localhost:8519). El dataset de demostración ya viene en `data/`; solo hace falta `python scripts/generate_dataset.py` si se quiere regenerar.

## Despliegue en Streamlit Community Cloud

El archivo principal (*Main file path*) es `app.py`. `run.py` es solo un lanzador local y no se usa en Cloud.

- `requirements.txt` fija las versiones de Python probadas; `packages.txt` instala `libgomp1`, que necesita XGBoost en Debian.
- `.streamlit/config.toml` solo define tema y opciones de cliente; el puerto y la dirección los pone Cloud.
- En *Advanced settings* elige Python 3.11 o 3.12.
- Cloud oculta los tracebacks: si aparece la pantalla "Oh no", el detalle está en los logs de *Manage app*.

## Módulos

1. **Análisis exploratorio.** Distribuciones, correlaciones, atípicos, mapa, precipitación, distancia a la Q. La bermejala y elevación.
2. **Machine learning.** Regresión logística, Random Forest y XGBoost. Accuracy, precision, recall, F1, ROC-AUC y matriz de confusión. El recall va primero.
3. **Predicción.** Ubicación, lluvia, elevación, pendiente, distancia a la Q. La bermejala y urbanización. Devuelve la probabilidad y el nivel.
4. **Recomendaciones.** Bajo (monitoreo normal), medio (seguimiento de lluvias), alto (evitar la margen) y muy alto (alertas oficiales).

## Dataset

Columnas de entrada:

`distancia_canada_m`, `distancia_drenaje_m`, `elevacion_msnm`, `diferencia_elevacion_m`, `pendiente_pct`, `precipitacion_1h_mm`, `precipitacion_6h_mm`, `precipitacion_24h_mm`, `precipitacion_3d_mm`, `precipitacion_7d_mm`, `humedad_suelo_pct`, `cobertura_vegetal_pct`, `superficie_impermeable_pct`, `densidad_construccion_pct`, `tipo_uso_suelo`, `estado_drenaje`, `historial_inundacion_eventos`.

Objetivo: `riesgo` (0 sin riesgo, 1 riesgo).

El archivo de demostración está en `data/Prevencion_del_riesgo.csv` y es el que usa la aplicación. Si se reemplaza por otro con las mismas columnas y viene de Excel en español, se aceptan separador `;` y decimales con coma.

`Images/mapa_hidrico.jpg` es el plano de referencia de la Q. La bermejala. Puede reemplazarse por otra imagen con el mismo nombre. `scripts/generate_dataset.py` dibuja un plano sintético aparte en `Images/mapa_hidrico_generado.png`, sin tocar el anterior.

## Aviso

Los sectores (Bermejal, Moravia, Berlín y los demás) y el trazo de la Q. La bermejala son una referencia de trabajo. No son la cartografía oficial del DAP ni una declaratoria de riesgo del barrio. Ante una emergencia, la fuente es el DAGRD, SIATA y la línea 123.

## Pruebas

```bash
pytest -q
```
