import matplotlib.pyplot as plt
import numpy as np
import random
import streamlit as st
import pandas as pd
import itertools
import pulp as p
import time

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

def se_intersectan2(l1,l2):
    if (l1['x0']==l1['x1']) and (l2['x0']==l2['x1']):
        if l1['x0']==l2['x0']:
            return (l1['y1']>l2['y0']) and (l1['y0']<l2['y1'])


def subconjuntos(lista_lineas):
    sub =[]
    for i in range(len(lista_lineas)+1):
        sub+= list(itertools.combinations(lista_lineas,i))
    return sub


def solutions_subsets_H(sub):
    sub_sorted = sub.sort_values(by='x1', ascending=True).reset_index(drop=True)
    sub_solution = pd.DataFrame({'x0':[] , 'y0': [],'x1':[],'y1':[]})
    for i in range(len(sub_sorted)):
        if i == 0:
            sub_solution = pd.concat([sub_solution, sub_sorted.iloc[[0]]], ignore_index=True)
        else:
            actual_solution = sub_solution.iloc[-1]

            l1x = [actual_solution['x0'], actual_solution['x1']]
            l1y = [actual_solution['y0'], actual_solution['y1']]

            l2x = [sub_sorted['x0'][i], sub_sorted['x1'][i]]
            l2y = [sub_sorted['y0'][i], sub_sorted['y1'][i]]

            if not se_intersectan(l1x,l1y,l2x,l2y):
                sub_solution = pd.concat([sub_solution, sub_sorted.iloc[[i]]])
    return sub_solution

def solutions_subsets_V(sub):
    sub_sorted = sub.sort_values(by='y1', ascending=True).reset_index(drop=True)
    sub_solution = pd.DataFrame({'x0':[] , 'y0': [],'x1':[],'y1':[]})
    for i in range(len(sub_sorted)):
        if i == 0:
            sub_solution = pd.concat([sub_solution, sub_sorted.iloc[[0]]], ignore_index=True)
        else:
            actual_solution = sub_solution.iloc[-1]

            l1x = [actual_solution['x0'], actual_solution['x1']]
            l1y = [actual_solution['y0'], actual_solution['y1']]

            l2x = [sub_sorted['x0'][i], sub_sorted['x1'][i]]
            l2y = [sub_sorted['y0'][i], sub_sorted['y1'][i]]

            if not se_intersectan(l1x,l1y,l2x,l2y):
                sub_solution = pd.concat([sub_solution, sub_sorted.iloc[[i]]])
    return sub_solution


st.markdown("# Testing")
st.sidebar.header("Experimental Enviroment")

if 'datatest' not in st.session_state:
    st.session_state.datatest = pd.DataFrame({'x':[] , 'y': []})
    st.session_state.dataworst = pd.DataFrame({'x':[] , 'y': []})
    st.session_state.datacomp = pd.DataFrame({'x':[] , 'y': []})
    st.session_state.datalinestest = []
    st.session_state.ctest = 0
    st.session_state.minmaxtest = [-10,10]
    st.session_state.linestest = []
    st.session_state.disabledtest = False

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
        porcentaje = calcular_porcentaje(parte,c)
        st.write("Number of Vertical Lines: " + str(porcentaje))
        st.write("Number of Horizontal Lines: "+ str(c-porcentaje))

col1, col2,col3,col4 = st.columns(4)
with col1:
    worst_case = st.button("Worst Case in 100 random cases")

with col2:
    time_test = st.button("Time experiment brute force")

with col3:
    button_lineal = st.button("Time experiment lineal solution", disabled=True)

with col4:
    button_comp = st.button("Comparacion Entera/Fraccionaria")    

if worst_case:
    worst = 1
    st.write("Comparacion OPT / 2-aprox")
    inicio_total = time.time()
    for i in range(100):
        st.session_state.ctest = c
        x=[]
        y=[]
        xoy=[]
        lines = []
        lineslist =[]
        check_lines =[]
        if not checkbox:
            for i in range(st.session_state.ctest):
                xoy.append(random.choice(["x","y"]))
        else:
            for i in range(st.session_state.ctest):
                if len(xoy)<porcentaje:
                    xoy.append("x")
                else:
                    xoy.append("y")
        for i in range(0,st.session_state.ctest):
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
        st.session_state.linestest = list(lines)
        st.session_state.datatest = pd.DataFrame(list(zip(lineslist,x, y)),columns =["Lines",'x', 'y'])
        st.session_state.datalinestest = pd.DataFrame(list(zip(lines, check_lines)),columns =['Lines', 'CheckBox'])
        st.session_state.minmaxtest = values

        #COMPARACION OPT/ALG
        
        modelo = p.LpProblem('Lineal_Solution', p.LpMaximize)

        X = p.LpVariable.dicts("Linea", st.session_state.linestest, lowBound=0, upBound=1, cat=p.LpInteger)
        modelo += p.lpSum(X[linea] for linea in st.session_state.linestest)
        for i in range(0,len(st.session_state.datatest["x"]),2):
            Li = st.session_state.datatest["Lines"][i]
            for j in range(i+2,len(st.session_state.datatest["x"]),2):
                Lj = st.session_state.datatest["Lines"][j]
                l1x = list(st.session_state.datatest["x"][i:i+2])
                l1y = list(st.session_state.datatest["y"][i:i+2])
                l2x = list(st.session_state.datatest["x"][j:j+2])
                l2y = list(st.session_state.datatest["y"][j:j+2])
                if se_intersectan(l1x,l1y,l2x,l2y):
                    modelo += X[Li] + X[Lj] <= 1, 'Restriccion de Intereseccion '+Li+'_'+Lj
        solucion = modelo.solve()
        OPT = int(p.value(modelo.objective))
        #st.write("Tamaño Máximo de la Solucion Por Programacion Lineal: ",str(OPT))
        
        Line_H = pd.DataFrame({'x0':[] , 'y0': [],'x1':[],'y1':[]})
        Line_V = pd.DataFrame({'x0':[] , 'y0': [],'x1':[],'y1':[]})
        for i in range(0,len(st.session_state.datatest["x"]),2):
            
            if st.session_state.datatest["x"][i]==st.session_state.datatest["x"][i+1]:
                Line_V.loc[len(Line_V)] = [st.session_state.datatest["x"][i], st.session_state.datatest["y"][i], st.session_state.datatest["x"][i+1], st.session_state.datatest["y"][i+1]]
            else:
                Line_H.loc[len(Line_H)] = [st.session_state.datatest["x"][i], st.session_state.datatest["y"][i], st.session_state.datatest["x"][i+1], st.session_state.datatest["y"][i+1]]

        # Crear un diccionario para almacenar los subconjuntos
        subsetsH = {}
        subsetsV = {}
        solution_H = pd.DataFrame({'x0':[] , 'y0': [],'x1':[],'y1':[]})
        solution_V = pd.DataFrame({'x0':[] , 'y0': [],'x1':[],'y1':[]})
        # Recorrer los grupos
        for y0_value, group in Line_H.groupby('y0'):
            if len(group) > 1:  # Verificar si el grupo tiene más de 1 fila
                subsetsH[y0_value] = group
            else:
                solution_H = pd.concat([solution_H, group])

        for y0_value, group in Line_V.groupby('x0'):
            if len(group) > 1:  # Verificar si el grupo tiene más de 1 fila
                subsetsV[y0_value] = group
            else:
                solution_V = pd.concat([solution_V, group])

        for item in subsetsH:
            solution_H = pd.concat([solution_H, solutions_subsets_H(subsetsH[item])]) 
        for item in subsetsV:
            solution_V = pd.concat([solution_V, solutions_subsets_V(subsetsV[item])]) 

        
        if len(solution_V)>len(solution_H):
            final = solution_V.reset_index(drop=True) 
        else:
            final = solution_H.reset_index(drop=True) 
        ALG = len(final)
        #st.write("Tamaño Solucion 2-APROX: "+ str(len(final)))

        if worst<OPT/ALG:
            worst = OPT/ALG
            st.session_state.dataworst = st.session_state.datatest
    tiempo_final = time.time() - inicio_total
    st.write("El peor caso de OPT/ALG en 100 casos aleatorios es:",worst)
    st.write("Calculado en:",tiempo_final,"segundos")
    st.dataframe(st.session_state.dataworst)
    fig, ax = plt.subplots()
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    for i in range(0, len(st.session_state.dataworst["x"]), 2):
            plt.plot(st.session_state.dataworst["x"][i:i+2], st.session_state.dataworst["y"][i:i+2], 'o-')


    plt.xlim(st.session_state.minmaxtest[0]-2, st.session_state.minmaxtest[1]+2)
    plt.ylim(st.session_state.minmaxtest[0]-2, st.session_state.minmaxtest[1]+2) 
    plt.minorticks_on()
    plt.grid( True, 'minor', markevery=2, linestyle='--' )
    plt.grid( True, 'major', markevery=10 )
    plt.title('Randomly Generated Lines Problem Worst Value')
    st.pyplot(fig)

if time_test:
    st.write("Time Test")
    # FUERZA BRUTA
    tiempo_brute = []
    
    for l in range(1,100):
        st.write("Iteracion:",l)
        st.session_state.ctest = l
        x=[]
        y=[]
        xoy=[]
        lines = []
        lineslist =[]
        check_lines =[]
        if not checkbox:
            for i in range(st.session_state.ctest):
                xoy.append(random.choice(["x","y"]))
        else:
            for i in range(st.session_state.ctest):
                if len(xoy)<porcentaje:
                    xoy.append("x")
                else:
                    xoy.append("y")
        for i in range(0,st.session_state.ctest):
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
        st.session_state.linestest = list(lines)
        st.session_state.datatest = pd.DataFrame(list(zip(lineslist,x, y)),columns =["Lines",'x', 'y'])
        st.session_state.datalinestest = pd.DataFrame(list(zip(lines, check_lines)),columns =['Lines', 'CheckBox'])
        st.session_state.minmaxtest = values

        tiempo_inicio = time.time()
        conjuntos = subconjuntos(st.session_state.linestest)
        Line_1=[]
        Line_2=[]
        Intersection =[]
        for i in range(0,len(st.session_state.datatest["x"]),2):
            for j in range(i+2,len(st.session_state.datatest["x"]),2):
                l1x = list(st.session_state.datatest["x"][i:i+2])
                l1y = list(st.session_state.datatest["y"][i:i+2])
                l2x = list(st.session_state.datatest["x"][j:j+2])
                l2y = list(st.session_state.datatest["y"][j:j+2])
                if se_intersectan(l1x,l1y,l2x,l2y):
                    Line_1.append("L"+str(i//2))
                    Line_2.append("L"+str(j//2))
                    Intersection.append(se_intersectan(l1x,l1y,l2x,l2y))
        conjunto_max = []
        tamaño_max = 0 
        num_optimas = 0      
        for i in range(len(conjuntos)):
            es_solucion = True
            if len(conjuntos[i])<2:
                conjunto_max = conjuntos[i]
                tamaño_max = len(conjuntos[i])
                continue
            for j in range(len(Line_1)):
                if Line_1[j] in conjuntos[i] and Line_2[j] in conjuntos[i]:
                    es_solucion = False
                    break
            if es_solucion:
                if tamaño_max == len(conjuntos[i]):
                    num_optimas+=1
                    continue
                conjunto_max = conjuntos[i]
                tamaño_max = len(conjuntos[i])
                num_optimas=1
                    
        tiempo_fin = time.time()
        if tiempo_fin-tiempo_inicio<=60:
            tiempo_brute.append(tiempo_fin-tiempo_inicio)
        else:
            st.write("Se excede el tiempo máximo permitido en el intento: ",len(tiempo_brute)+1)
            break

    
    st.write("Numero de lineas Fuerza Bruta:", len(tiempo_brute))
    st.write(tiempo_brute)

    



if button_lineal:
    st.write("Time Test")
    time_lineal =[]
    for l in range(1000,10000):
        st.session_state.ctest = l
        x=[]
        y=[]
        xoy=[]
        lines = []
        lineslist =[]
        check_lines =[]
        if not checkbox:
            for i in range(st.session_state.ctest):
                xoy.append(random.choice(["x","y"]))
        else:
            for i in range(st.session_state.ctest):
                if len(xoy)<porcentaje:
                    xoy.append("x")
                else:
                    xoy.append("y")
        for i in range(0,st.session_state.ctest):
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
        st.session_state.linestest = list(lines)
        st.session_state.datatest = pd.DataFrame(list(zip(lineslist,x, y)),columns =["Lines",'x', 'y'])
        st.session_state.datalinestest = pd.DataFrame(list(zip(lines, check_lines)),columns =['Lines', 'CheckBox'])
        st.session_state.minmaxtest = values

        modelo = p.LpProblem('Lineal_Solution', p.LpMaximize)

        X = p.LpVariable.dicts("Linea", st.session_state.linestest, lowBound=0, upBound=1, cat=p.LpInteger)
        modelo += p.lpSum(X[linea] for linea in st.session_state.linestest)
        for i in range(0,len(st.session_state.datatest["x"]),2):
            Li = st.session_state.datatest["Lines"][i]
            for j in range(i+2,len(st.session_state.datatest["x"]),2):
                Lj = st.session_state.datatest["Lines"][j]
                l1x = list(st.session_state.datatest["x"][i:i+2])
                l1y = list(st.session_state.datatest["y"][i:i+2])
                l2x = list(st.session_state.datatest["x"][j:j+2])
                l2y = list(st.session_state.datatest["y"][j:j+2])
                if se_intersectan(l1x,l1y,l2x,l2y):
                    modelo += X[Li] + X[Lj] <= 1, 'Restriccion de Intereseccion '+Li+'_'+Lj
        solucion = modelo.solve()

        if modelo.solutionTime<=60:
            time_lineal.append(modelo.solutionTime)
            st.write("iteracion:", l )
        else:
            st.write("Se excede el tiempo máximo permitido en el intento: ",l+1)
            break
    
    st.write("Numero de lineas Solucion Lineal:", len(time_lineal))
    st.write(time_lineal)
    
if button_comp:
    worst = 0
    st.write("Comparacion OPT  ENTERO/FRACC")
    inicio_total = time.time()
    for i in range(100):
        st.session_state.ctest = c
        x=[]
        y=[]
        xoy=[]
        lines = []
        lineslist =[]
        check_lines =[]
        if not checkbox:
            for i in range(st.session_state.ctest):
                xoy.append(random.choice(["x","y"]))
        else:
            for i in range(st.session_state.ctest):
                if len(xoy)<porcentaje:
                    xoy.append("x")
                else:
                    xoy.append("y")
        for i in range(0,st.session_state.ctest):
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
        st.session_state.linestest = list(lines)
        st.session_state.datatest = pd.DataFrame(list(zip(lineslist,x, y)),columns =["Lines",'x', 'y'])
        st.session_state.datalinestest = pd.DataFrame(list(zip(lines, check_lines)),columns =['Lines', 'CheckBox'])
        st.session_state.minmaxtest = values

        #COMPARACION OPT/ALG
        
        modelo = p.LpProblem('Lineal_Solution', p.LpMaximize)

        X = p.LpVariable.dicts("Linea", st.session_state.linestest, lowBound=0, upBound=1, cat=p.LpInteger)
        modelo += p.lpSum(X[linea] for linea in st.session_state.linestest)
        for i in range(0,len(st.session_state.datatest["x"]),2):
            Li = st.session_state.datatest["Lines"][i]
            for j in range(i+2,len(st.session_state.datatest["x"]),2):
                Lj = st.session_state.datatest["Lines"][j]
                l1x = list(st.session_state.datatest["x"][i:i+2])
                l1y = list(st.session_state.datatest["y"][i:i+2])
                l2x = list(st.session_state.datatest["x"][j:j+2])
                l2y = list(st.session_state.datatest["y"][j:j+2])
                if se_intersectan(l1x,l1y,l2x,l2y):
                    modelo += X[Li] + X[Lj] <= 1, 'Restriccion de Intereseccion '+Li+'_'+Lj
        solucion = modelo.solve()
        OPT = int(p.value(modelo.objective))
        #st.write("Tamaño Máximo de la Solucion Por Programacion Lineal: ",str(OPT))
        
        modelo2 = p.LpProblem('Lineal_Solution', p.LpMaximize)

        X = p.LpVariable.dicts("Linea", st.session_state.linestest, lowBound=0, upBound=1, cat='Continuous')
        modelo2 += p.lpSum(X[linea] for linea in st.session_state.linestest)
        for i in range(0,len(st.session_state.datatest["x"]),2):
            Li = st.session_state.datatest["Lines"][i]
            for j in range(i+2,len(st.session_state.datatest["x"]),2):
                Lj = st.session_state.datatest["Lines"][j]
                l1x = list(st.session_state.datatest["x"][i:i+2])
                l1y = list(st.session_state.datatest["y"][i:i+2])
                l2x = list(st.session_state.datatest["x"][j:j+2])
                l2y = list(st.session_state.datatest["y"][j:j+2])
                if se_intersectan(l1x,l1y,l2x,l2y):
                    modelo2 += X[Li] + X[Lj] <= 1, 'Restriccion de Intereseccion '+Li+'_'+Lj
        solucion = modelo2.solve()
        ALG = int(p.value(modelo2.objective))


        if worst<ALG/OPT:
            worst = ALG/OPT
            st.session_state.datacomp = st.session_state.datatest
            lista_solucion =[]
            lista_name =[]
            for v in modelo2.variables():
                lista_solucion.append(float(v.varValue))
                lista_name.append(v.name.replace("Linea_",""))
            final_solution =pd.DataFrame(list(zip(lista_name, lista_solucion)),columns =['Lines', 'Solution'])
            lista_solucion =[]
            lista_name =[]
            for v in modelo.variables():
                lista_solucion.append(int(v.varValue))
                lista_name.append(v.name.replace("Linea_",""))
            final_solution2 =pd.DataFrame(list(zip(lista_name, lista_solucion)),columns =['Lines', 'Solution'])
            

    tiempo_final = time.time() - inicio_total
    st.write("El peor caso de ENTERO/FRACC en 100 casos aleatorios es:",worst)
    st.write("Calculado en:",tiempo_final,"segundos")
    st.dataframe(st.session_state.datacomp)
    st.dataframe(final_solution)
    st.dataframe(final_solution2)
    fig, ax = plt.subplots()
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    for i in range(0, len(st.session_state.datacomp["x"]), 2):
            plt.plot(st.session_state.datacomp["x"][i:i+2], st.session_state.datacomp["y"][i:i+2], 'o-')


    plt.xlim(st.session_state.minmaxtest[0]-2, st.session_state.minmaxtest[1]+2)
    plt.ylim(st.session_state.minmaxtest[0]-2, st.session_state.minmaxtest[1]+2) 
    plt.minorticks_on()
    plt.grid( True, 'minor', markevery=2, linestyle='--' )
    plt.grid( True, 'major', markevery=10 )
    plt.title('Randomly Generated Lines Problem Worst Comparative Value')
    st.pyplot(fig)
