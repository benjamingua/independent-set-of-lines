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
with st.container(border=True):
    values = st.slider("Select a range of the axes (square cartesian axis)", -100.0, 100.0,(-20.0,20.0),step=0.5)
    st.write("Values:", values)
    rangoxy = np.arange(values[0],values[1],0.5)

    c = st.slider("Number of Lines", 0, 100, 10)

    if st.button("Let's graph"):
        x=[]
        y=[]

        for i in range(0,c):
            xoy = random.choice(["x","y"])
            temporal = random.choice(rangoxy)
            if xoy ==  "x":
                x.append(temporal)
                x.append(temporal)
                y.append(random.choice(rangoxy))
                y.append(random.choice(rangoxy))
            else:            
                x.append(random.choice(rangoxy))
                x.append(random.choice(rangoxy))
                y.append(temporal)
                y.append(temporal)
        dataframe = pd.DataFrame(list(zip(x, y)),columns =['x', 'y'])
        st.write("Dataframe preview")
        st.dataframe(dataframe,hide_index=True,use_container_width=True)
        fig, ax = plt.subplots()
        for i in range(0, len(x), 2):
            plt.plot(x[i:i+2], y[i:i+2], 'o-')
            plt.annotate( 'L'+str(i//2), PuntoMedio(x[i:i+2], y[i:i+2]), color='blue' )

        plt.axhline(0,color='black')
        plt.axvline(0,color='black')

        plt.minorticks_on()
        plt.grid( True, 'minor', markevery=2, linestyle='--' )
        plt.grid( True, 'major', markevery=10 )
        plt.title('Randomly Generated Lines Problem')
        st.pyplot(fig)