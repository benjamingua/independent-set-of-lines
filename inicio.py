import streamlit as st



st.set_page_config(
    page_title="Inicio",
    page_icon="👋",
)
st.sidebar.success("Select a Page.")
st.markdown("""
            <h1>
                <div style="text-align: center;">Proyecto visualización e implementación de algoritmos de conjunto independiente para líneas.</div>
            </h1>
""", unsafe_allow_html=True
)
st.markdown(
    """
    <div style="text-align: justify;">
        - El objetivo de esta aplicación web es brindar apoyo a la investigación de problema de conjunto independiente para líneas
        mediante la visualización del problema y su solución implementada con un algoritmo glotón para lineas generadas aleatoriamente
        y también para conjuntos de lineas definidas en archivos CSV. 
    </div>

    <div style="text-align: justify;">
    - Random Lines permite crear un conjunto de lineas generadas aleatoriamente eligiendo previamente
    algunos parametros para el estudio y grafica en el plano dichas lineas y su solución.
    </div>

    <div style="text-align: justify;">
    - Upload Lines recibe un archivo CSV que contiene las coordenadas de los puntos que forman las líneas y 
    grafica en el plano dichas lineas y su solución.
    </div>
    
""", unsafe_allow_html=True
)