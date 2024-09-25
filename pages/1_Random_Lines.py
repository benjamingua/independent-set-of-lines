import matplotlib.pyplot as plt
import numpy as np
import random
import streamlit as st
import pandas as pd

def PuntoMedio(x,y):
    if x[0]==x[1]:
        return (x[0]+x[1])/2+0.1, (y[0]+y[1])/2-0.3
    else:
        return (x[0]+x[1])/2-0.4, (y[0]+y[1])/2+0.3

def rangominmax(value,rango,sizeline):
    menor = sizeline[0]
    mayor = sizeline[1]
    resultados = [x for x in rango if menor <= np.abs(x-value)<= mayor]
    return resultados

def viable_point(rango,sizeline):
    viable =[]
    for i in rango:
        if i+sizeline[0]<=rango[-1] or i-sizeline[0]>=rango[0]:
            viable.append(i)
    return viable

def calcular_porcentaje(percent, total):
    porcentaje = (percent * total) // 100
    return porcentaje

st.markdown("# Random Lines")
st.sidebar.header("Random Lines")
st.markdown(
    """
    **Instructions:** 
    - Select parameters
    - Let's graph

    **OBS:** 
    - You can download the randomly generated lines as a CSV file by clicking the download icon in the dataframe preview.
    - If any parameter is changed or press Let's graph again, the previous set of lines will instantly disappear.

    
    """
)
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = pd.DataFrame({'x':[] , 'y': []})
    st.session_state.datalines = []
    st.session_state.c = 0
    st.session_state.minmax = [-10,10]


with st.container(border=True):
    values = st.slider("Select a range of the axes (square cartesian axis)", -100.0, 100.0,(-20.0,20.0),step=0.5)
    dif = values[1]-values[0]
    minmax = st.slider("Select the minimum and maximum size of the lines.)", 0.0, dif,(0.0,dif),step=0.5)
    rangoxy = np.arange(values[0],values[1]+0.5,0.5)
    possible = viable_point(rangoxy,minmax)
    c = st.slider("Number of Lines", 0, 100, 10)
    checkbox = st.checkbox(" % of vertical and horizontal lines.",key="percent")
    if checkbox:
        parte = st.slider("% of Lines Vertical", 0, 100, step=10)
        porcentaje = calcular_porcentaje(parte,st.session_state.c)
        st.write("Number of Vertical Lines: " + str(porcentaje))
        st.write("Number of Horizontal Lines: "+ str(st.session_state.c-porcentaje))

    if st.button("Create Random Lines Set"):
        st.session_state.c = c
        x=[]
        y=[]
        xoy=[]
        lines = []
        check_lines =[]
        if not checkbox:
            for i in range(st.session_state.c):
                xoy.append(random.choice(["x","y"]))
        else:
            for i in range(st.session_state.c):
                if len(xoy)<porcentaje:
                    xoy.append("x")
                else:
                    xoy.append("y")
        for i in range(0,st.session_state.c):
            if st.session_state.c<=20:
                lines.append("L"+str(i))
                check_lines.append(True)
            temporal = random.choice(rangoxy)
            initial_point = random.choice(possible)
            temporal_range = rangominmax(initial_point,rangoxy,minmax)
            if xoy[i] ==  "x":
                x.append(temporal)
                x.append(temporal)
                y.append(initial_point)
                y.append(random.choice(temporal_range))
            else:            
                x.append(initial_point)
                x.append(random.choice(temporal_range))
                y.append(temporal)
                y.append(temporal)
        st.session_state.dataframe = pd.DataFrame(list(zip(x, y)),columns =['x', 'y'])
        st.session_state.datalines = pd.DataFrame(list(zip(lines, check_lines)),columns =['Lines', 'CheckBox'])
        st.session_state.minmax = values

with st.container():
        st.write("Dataframe preview")
        st.dataframe(st.session_state.dataframe,hide_index=True,use_container_width=True)
        if st.session_state.c<=20:
            col1, col2 = st.columns([0.75, 0.25])
            with col2:
                st.write("Dataframe Lines Draw")
                edited = st.data_editor(st.session_state.datalines,hide_index=True,use_container_width=True)
        else:
            col1,col2 = st.columns([0.99,0.01])
        fig, ax = plt.subplots()
        plt.axhline(0,color='black')
        plt.axvline(0,color='black')
        for i in range(0, len(st.session_state.dataframe["x"]), 2):
            if st.session_state.c<=20 and edited["CheckBox"][i//2]:               
                plt.plot(st.session_state.dataframe["x"][i:i+2], st.session_state.dataframe["y"][i:i+2], 'o-')
                plt.annotate( 'L'+str(i//2), PuntoMedio(list(st.session_state.dataframe["x"][i:i+2]), list(st.session_state.dataframe["y"][i:i+2])), color='blue' )
            elif st.session_state.c> 20:
                plt.plot(st.session_state.dataframe["x"][i:i+2], st.session_state.dataframe["y"][i:i+2], 'o-')

        plt.xlim(st.session_state.minmax[0], st.session_state.minmax[1])
        plt.ylim(st.session_state.minmax[0], st.session_state.minmax[1]) 
        plt.minorticks_on()
        plt.grid( True, 'minor', markevery=2, linestyle='--' )
        plt.grid( True, 'major', markevery=10 )
        plt.title('Randomly Generated Lines Problem')
        col1.pyplot(fig)
