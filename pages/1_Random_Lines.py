import matplotlib.pyplot as plt
import numpy as np
import random
import streamlit as st
import pandas as pd
import itertools

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


def se_intersectan(l1x,l1y,l2x,l2y):
    if (l1x[0]==l1x[1]) and (l2x[0] == l2x[1]):
        if l1x[0]==l2x[0]:
            return (l1y[1]>l2y[0]) and (l1y[0]<l2y[1])
        else:
            return False
    if (l1y[0] == l2y[0]) and (l2y[0] == l2y[1]):
        if l1y[0]==l2y[0]:
            return (l1x[1]>l2x[0]) and (l1x[0]<l2x[1])
        else:
            return False
    if l1y[0] == l1y[1]:
        return (l2y[0]<l1y[0]<l2y[1]) and (l1x[0]<l2x[0]<l1x[1])
    else:
        return (l2x[0]<l1x[0]<l2x[1]) and (l1y[0]<l2y[0]<l1y[1]) 

def subconjuntos(lista_lineas):
    sub =[]
    for i in range(len(lista_lineas)+1):
        sub+= list(itertools.combinations(lista_lineas,i))
    return sub


def alg_d1x(lineas):
    lineas_l = lineas[::2]
    lineas_r = lineas[1::2]
    return

    
st. set_page_config(layout="wide")
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
    st.session_state.lines = []


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
        lineslist =[]
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
            lines.append("L"+str(i))
            lineslist.append("L"+str(i))
            lineslist.append("L"+str(i))
            check_lines.append(True)
            temporal = random.choice(rangoxy)
            initial_point = random.choice(possible)
            temporal_range = rangominmax(initial_point,rangoxy,minmax)
            if xoy[i] ==  "x":
                x.append(temporal)
                x.append(temporal)
                point = random.choice(temporal_range)
                if point>=initial_point:
                    y.append(initial_point)
                    y.append(point)
                else:
                    y.append(point)
                    y.append(initial_point)
            else:  
                point = random.choice(temporal_range)     
                if point >= initial_point:
                    x.append(initial_point)
                    x.append(point)
                else:
                    x.append(point)
                    x.append(initial_point)
                y.append(temporal)
                y.append(temporal)
        st.session_state.lines = list(lines)
        st.session_state.dataframe = pd.DataFrame(list(zip(lineslist,x, y)),columns =["Lines",'x', 'y'])
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

        plt.xlim(st.session_state.minmax[0]-2, st.session_state.minmax[1]+2)
        plt.ylim(st.session_state.minmax[0]-2, st.session_state.minmax[1]+2) 
        plt.minorticks_on()
        plt.grid( True, 'minor', markevery=2, linestyle='--' )
        plt.grid( True, 'major', markevery=10 )
        plt.title('Randomly Generated Lines Problem')
        col1.pyplot(fig)
        Line_1=[]
        Line_2=[]
        Intersection =[]
        for i in range(0,len(st.session_state.dataframe["x"]),2):
            for j in range(i+2,len(st.session_state.dataframe["x"]),2):
                l1x = list(st.session_state.dataframe["x"][i:i+2])
                l1y = list(st.session_state.dataframe["y"][i:i+2])
                l2x = list(st.session_state.dataframe["x"][j:j+2])
                l2y = list(st.session_state.dataframe["y"][j:j+2])
                if se_intersectan(l1x,l1y,l2x,l2y):
                    Line_1.append("L"+str(i//2))
                    Line_2.append("L"+str(j//2))
                    Intersection.append(se_intersectan(l1x,l1y,l2x,l2y))
        intersecciones = pd.DataFrame({'Line 1':Line_1 , 'Line 2': Line_2, "Intersection":Intersection})        
        st.dataframe(intersecciones,hide_index=True,use_container_width=True)
        if st.button("Generar Solucion"):
            st.session_state.conjuntos = subconjuntos(st.session_state.lines)
            st.write("La cantidad de subconjuntos es:" + str(len(st.session_state.conjuntos)))
        
            conjunto_max = []
            tamaño_max = 0       
            for i in range(len(st.session_state.conjuntos)):
                es_solucion = True
                if len(st.session_state.conjuntos[i])<2:
                    conjunto_max = st.session_state.conjuntos[i]
                    tamaño_max = len(st.session_state.conjuntos[i])
                    continue
                for j in range(len(Line_1)):
                    if Line_1[j] in st.session_state.conjuntos[i] and Line_2[j] in st.session_state.conjuntos[i]:
                        es_solucion = False
                        break
                if es_solucion:
                    conjunto_max = st.session_state.conjuntos[i]
                    tamaño_max = len(st.session_state.conjuntos[i])

            st.write("El conjunto máximo es: ", conjunto_max)
            st.write("De tamaño "+ str(tamaño_max))

            fig2, ax = plt.subplots()
            plt.axhline(0,color='black')
            plt.axvline(0,color='black')
            for i in range(0, len(st.session_state.dataframe["x"]), 2):
                if st.session_state.dataframe["Lines"][i] in conjunto_max:  
                    plt.plot(st.session_state.dataframe["x"][i:i+2], st.session_state.dataframe["y"][i:i+2], 'o-')
                    plt.annotate( 'L'+str(i//2), PuntoMedio(list(st.session_state.dataframe["x"][i:i+2]), list(st.session_state.dataframe["y"][i:i+2])), color='blue' )
  
            plt.xlim(st.session_state.minmax[0]-2, st.session_state.minmax[1]+2)
            plt.ylim(st.session_state.minmax[0]-2, st.session_state.minmax[1]+2) 
            plt.minorticks_on()
            plt.grid( True, 'minor', markevery=2, linestyle='--' )
            plt.grid( True, 'major', markevery=10 )
            plt.title('Randomly Generated Lines Problem Solution')
            st.pyplot(fig2)
        
