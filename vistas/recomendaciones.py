import streamlit as st

from src.recommendations import LEVELS
from src.ui import disclaimer, page_header

page_header(
    "Módulo 4",
    "Qué hacer con el resultado",
    "La probabilidad no se entrega sola. Cada franja trae una conducta. El consejo cambia si el punto está en la margen, si el drenaje no da abasto o si la lluvia de la hora ya es intensa.",
)

st.markdown(
    """
    | Nivel | Probabilidad | Recomendación |
    | --- | --- | --- |
    | Bajo | Menos de 25 % | Monitoreo normal |
    | Medio | 25 % a 50 % | Seguimiento de lluvias |
    | Alto | 50 % a 75 % | Evitar actividades cercanas al cauce |
    | Muy alto | 75 % o más | Priorizar medidas preventivas y atención a alertas oficiales |
    """
)

cols = st.columns(2)
for index, level in enumerate(LEVELS):
    with cols[index % 2]:
        actions = "".join(f"<li>{action}</li>" for action in level.actions)
        st.markdown(
            f"""
            <div class="level-card" style="--tone:{level.color}">
              <p class="kicker">{level.name}</p>
              <h3>{level.title}</h3>
              <p>{level.summary}</p>
              <ul>{actions}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.subheader("Cuando el punto tiene una condición concreta")
st.markdown(
    """
    - **Lluvia de la última hora de 20 mm o más.** La creciente de la Q. La bermejala puede llegar en minutos. No se trata de esperar el parte de la noche.
    - **A 30 m o menos del eje.** El consejo sube de tono: hay que salir de esa franja, no solo mirar el pronóstico.
    - **A 3 m o menos por encima del cauce.** Una creciente corta alcanza el punto aunque la lluvia no sea la peor del año.
    - **Drenaje deficiente o colapsado.** No cuente con la alcantarilla para sacar el agua de la calle.
    - **Dos o más inundaciones en el historial.** Use la ruta que el sector ya conoce.
    - **Superficie impermeable alta o suelo ya húmedo.** El agua corre en vez de infiltrarse, y llega antes a la Q. La bermejala.
    - **Pendiente fuerte y poca distancia.** El material que arrastra el cauce gana velocidad.
    """
)

st.subheader("A quién hay que creerle")
st.markdown(
    """
    Si el nivel sale alto o muy alto, el paso siguiente es una fuente oficial, no otra pasada por este modelo.

    - [SIATA](https://siata.gov.co/), el sistema de alerta temprana de Medellín y el Valle de Aburrá.
    - Departamento Administrativo de Gestión del Riesgo de Desastres, DAGRD, de la Alcaldía de Medellín.
    - Línea de emergencias **123**.

    Los sectores que ofrece la predicción —Bermejal, Moravia, Berlín y los demás— sirven para situar el ejercicio.
    Sus números no son una declaratoria de riesgo del barrio.
    """
)

st.page_link("vistas/prediccion.py", label="Volver a consultar un punto")
st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
disclaimer()
