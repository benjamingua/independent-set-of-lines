import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import random
import itertools
import pulp as p


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


def order_rectangle(rectangle):
    if rectangle["x"][0]>rectangle["x"][2]:
        if rectangle["y"][0]>rectangle["y"][1]:
            RU = [rectangle["x"][0], rectangle["y"][0]]
            RD = [rectangle["x"][1], rectangle["y"][1]]
            LU = [rectangle["x"][3], rectangle["y"][3]]
            LD = [rectangle["x"][2], rectangle["y"][2]]
        else:
            RU = [rectangle["x"][1], rectangle["y"][1]]
            RD = [rectangle["x"][0], rectangle["y"][0]]
            LU = [rectangle["x"][2], rectangle["y"][2]]
            LD = [rectangle["x"][3], rectangle["y"][3]]
    else:
        if rectangle["y"][0]>rectangle["y"][1]:
            RU = [rectangle["x"][3], rectangle["y"][3]]
            RD = [rectangle["x"][2], rectangle["y"][2]]
            LU = [rectangle["x"][0], rectangle["y"][0]]
            LD = [rectangle["x"][1], rectangle["y"][1]]
        else:
            RU = [rectangle["x"][2], rectangle["y"][2]]
            RD = [rectangle["x"][3], rectangle["y"][3]]
            LU = [rectangle["x"][1], rectangle["y"][1]]
            LD = [rectangle["x"][0], rectangle["y"][0]]
    return [LU,RU,RD,LD]


def interseccion(rec1, rec2):
    if rec2[0][0]<rec1[0][0] and rec2[1][0]>rec1[1][0] and rec1[0][1]> rec2[0][1] and rec1[3][1]<rec2[3][1]:
        return True
    if rec1[0][0]<rec2[0][0] and rec1[1][0]>rec2[1][0] and rec2[0][1]> rec1[0][1] and rec2[3][1]<rec1[3][1]:
        return True
    if chequear_esquinas0(rec1[0],rec2):
        return True
    if chequear_esquinas1(rec1[1],rec2):
        return True
    if chequear_esquinas2(rec1[2],rec2):
        return True
    if chequear_esquinas3(rec1[3],rec2):
        return True
    if chequear_esquinas0(rec2[0],rec1):
        return True
    if chequear_esquinas1(rec2[1],rec1):
        return True
    if chequear_esquinas2(rec2[2],rec1):
        return True
    if chequear_esquinas3(rec2[3],rec1):
        return True
    return False

def chequear_esquinas0(esquina,rec):
    return rec[0][0]<=esquina[0]<rec[1][0] and rec[3][1]<esquina[1]<=rec[0][1]
def chequear_esquinas1(esquina,rec):
    return rec[0][0]<esquina[0]<=rec[1][0] and rec[3][1]<esquina[1]<=rec[0][1]
def chequear_esquinas2(esquina,rec):
    return rec[0][0]<esquina[0]<=rec[1][0] and rec[3][1]<=esquina[1]<rec[0][1]
def chequear_esquinas3(esquina,rec):
    return rec[0][0]<=esquina[0]<rec[1][0] and rec[3][1]<=esquina[1]<rec[0][1]

def subconjuntos(lista_rec):
    sub =[]
    for i in range(len(lista_rec)+1):
        sub+= list(itertools.combinations(lista_rec,i))
    return sub
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

if 'datarec' not in st.session_state:
    st.session_state.datarec = pd.DataFrame({'x':[] , 'y': []})


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
        r=[]
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
            r.append("R"+str(i))
            #second point
            x.append(temporal)
            y.append(ran)
            r.append("R"+str(i))
            #third point
            x.append(ran2)
            y.append(ran)
            r.append("R"+str(i))
            #fourth point
            x.append(ran2)
            y.append(initial_point)
            r.append("R"+str(i))


        st.session_state.datarec = pd.DataFrame(list(zip(x, y, r)),columns =['x', 'y', 'rectangle'])
    st.write("Dataframe preview")
    st.dataframe(st.session_state.datarec,hide_index=True,use_container_width=True)
    fig, ax = plt.subplots()
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    rectangles = dict()
    for i in range(0, len(st.session_state.datarec["x"]), 4):
        graph = st.session_state.datarec.iloc[i:i+4].reset_index()
        name = "R"+str(i//4)
        rectangles[name] = order_rectangle(graph)
        graph = pd.concat([graph,st.session_state.datarec.iloc[i:i+1]], ignore_index=True)
        plt.plot(graph["x"],graph["y"], 'o-')


    plt.minorticks_on()
    plt.grid( True, 'minor', markevery=2, linestyle='--' )
    plt.grid( True, 'major', markevery=10 )
    plt.title('Random Rectangles Problem')
    st.pyplot(fig)
    lista =list(rectangles)
    st.write(lista)
    

    col1, col2= st.columns(2)
    with col1:
        button_brute = st.button("Generar Solucion Fuerza Bruta")

    with col2:
        button_lineal = st.button("Generar Solucion Lineal")

    if button_brute:
        conjuntos = subconjuntos(lista)
        Rec_1=[]
        Rec_2=[]
        Intersection =[]
        conjunto_max = []
        tamaño_max = 0 
        num_optimas = 0
        for i in range(0,len(lista)):
            for j in range(i+1, len(lista)):
                rec1 = rectangles[lista[i]]
                rec2 = rectangles[lista[j]]
                if interseccion(rec1,rec2):
                    Rec_1.append(lista[i])
                    Rec_2.append(lista[j])
                    Intersection.append(True)

                #st.write(lista[i],lista[j])
                #st.write(interseccion(rec1,rec2)) 

        for i in range(len(conjuntos)):
            es_solucion = True
            if len(conjuntos[i])<2:
                conjunto_max = conjuntos[i]
                tamaño_max = len(conjuntos[i])
                continue
            for j in range(len(Rec_1)):
                if Rec_1[j] in conjuntos[i] and Rec_2[j] in conjuntos[i]:
                    es_solucion = False
                    break
            if es_solucion:
                if tamaño_max == len(conjuntos[i]):
                    num_optimas+=1
                    continue
                conjunto_max = conjuntos[i]
                tamaño_max = len(conjuntos[i])
                num_optimas=1
                
        st.write("Existen "+str(num_optimas)+" soluciones optimas")
        st.write("El conjunto máximo es: ", conjunto_max)
        st.write("De tamaño "+ str(tamaño_max))

        fig1, ax = plt.subplots()
        plt.axhline(0,color='black')
        plt.axvline(0,color='black')
        rectangles = dict()
        for i in range(0, len(st.session_state.datarec["x"]), 4):
            if st.session_state.datarec["rectangle"][i] in conjunto_max: 
                graph = st.session_state.datarec.iloc[i:i+4].reset_index()
                graph = pd.concat([graph,st.session_state.datarec.iloc[i:i+1]], ignore_index=True)
                plt.plot(graph["x"],graph["y"], 'o-')

        plt.minorticks_on()
        plt.grid( True, 'minor', markevery=2, linestyle='--' )
        plt.grid( True, 'major', markevery=10 )
        plt.title('Random Rectangles Problem Brute Solution')
        st.pyplot(fig1)



    if button_lineal:
        modelo = p.LpProblem('Lineal_Solution', p.LpMaximize)
        X = p.LpVariable.dicts("Rectangle", rectangles, lowBound=0, upBound=1, cat=p.LpInteger)
        modelo += p.lpSum(X[rec] for rec in rectangles)
        for i in range(0,len(lista)):
            Li = lista[i]
            for j in range(i+1, len(lista)):
                Lj = lista[j]
                rec1 = rectangles[lista[i]]
                rec2 = rectangles[lista[j]]
                if interseccion(rec1,rec2):
                    modelo += X[Li] + X[Lj] <= 1, 'Restriccion de Intereseccion '+Li+'_'+Lj
        solucion = modelo.solve()
        st.write("Tamaño Máximo de la Solucion: ",str(int(p.value(modelo.objective))))
        lista_solucion =[]
        lista_name =[]
        for v in modelo.variables():
            lista_solucion.append(int(v.varValue))
            lista_name.append(v.name.replace("Rectangle_",""))
        final_solution =pd.DataFrame(list(zip(lista_name, lista_solucion)),columns =['Lines', 'Solution'])
        st.dataframe(final_solution)