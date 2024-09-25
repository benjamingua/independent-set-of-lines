import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import random

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

st.markdown("# Rectangle Experimental Upload")
st.sidebar.header("Rectangle Experimental Upload")
st.markdown(
    """
    **Instructions:** 

    - Select parameters
    - Let's graph

    **OBS:** 
    - You can download the randomly generated rectangles as a CSV file by clicking the download icon in the dataframe preview.
    - If any parameter is changed or press Let's graph again, the previous set of rectangles will instantly disappear.


    
    """
)
with st.container(border=True):
    values = st.slider("Select a range of the axes (square cartesian axis)", -100.0, 100.0,(-20.0,20.0),step=0.5)
    dif = values[1]-values[0]
    minmax = st.slider("Select the minimum and maximum size of the sides of the rectangle.)", 0.0, dif,(0.0,dif),step=0.5)
    rangoxy = np.arange(values[0],values[1]+0.5,0.5)
    possible = viable_point(rangoxy,minmax)
    c = st.slider("Number of Rectangles", 0, 20, 1)


    if st.button("Let's graph"):
        x=[]
        y=[]
        for i in range(0,c):
            temporal = random.choice(possible)
            initial_point = random.choice(possible)
            temporal_range = rangominmax(initial_point,rangoxy,minmax)
            ran = random.choice(temporal_range)
            temporal_range2 = rangominmax(temporal,rangoxy,minmax)
            ran2 = random.choice(temporal_range2)
            #first point
            x.append(temporal)
            y.append(initial_point)
            #second point
            x.append(temporal)
            y.append(ran)
            #third point
            x.append(ran2)
            y.append(ran)
            #fourth point
            x.append(ran2)
            y.append(initial_point)


        dataframe = pd.DataFrame(list(zip(x, y)),columns =['x', 'y'])
        st.write("Dataframe preview")
        st.dataframe(dataframe,hide_index=True,use_container_width=True)
        fig, ax = plt.subplots()
        plt.axhline(0,color='black')
        plt.axvline(0,color='black')
        for i in range(0, len(dataframe["x"]), 4):
            graph = dataframe.iloc[i:i+4]
            graph = pd.concat([graph,dataframe.iloc[i:i+1]], ignore_index=True)
            plt.plot(graph["x"],graph["y"], 'o-')


        plt.minorticks_on()
        plt.grid( True, 'minor', markevery=2, linestyle='--' )
        plt.grid( True, 'major', markevery=10 )
        plt.title('Upload Lines Problem')
        st.pyplot(fig)