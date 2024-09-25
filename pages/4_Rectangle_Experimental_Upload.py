import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

def PuntoMedio(x,y):
    if x[0]==x[1]:
        return (x[0]+x[1])/2+0.1, (y[0]+y[1])/2-0.3
    else:
        return (x[0]+x[1])/2-0.4, (y[0]+y[1])/2+0.3

st.markdown("# Rectangle Experimental Upload")
st.sidebar.header("Rectangle Experimental Upload")
st.markdown(
    """
    **Instructions:** 
    - Upload a CSV File  
    - First Column X Coordinate  
    - Second Column Y Coordinate
    
    """
)
with st.container(border=True):
    separator =st.selectbox("Please select CSV separator",[";",","])
    uploaded_file = st.file_uploader("Choose a file",type="CSV")

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file,sep=separator)
        st.write("Dataframe preview")
        st.dataframe(dataframe,hide_index=True,use_container_width=True)

    if st.button("Let's graph"):
#        try:
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
#        except:
#            st.error('No file uploaded or the file is not in the correct format', icon="🚨")
