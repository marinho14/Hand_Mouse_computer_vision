##Importacion de librerias
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



#
# sig_ok= pd.read_csv('click_ok.csv')
# sig_ok2=pd.read_csv('click_ok2.csv')
#
# A= sig_ok2.shape[0]
# B= sig_ok.shape[0]
# if len(sig_ok)>len(sig_ok2):
#     sig_ok= sig_ok.drop(range(A,B),axis=0)
#
# else:
#     sig_ok2 = sig_ok2.drop(range(B, A), axis=0)
#
# sig_t= pd.concat([sig_ok,sig_ok2],axis=0)
# X_train, X_test, y_train, y_test = train_test_split(sig_t.iloc[:,:-1],sig_t.iloc[:,-1], random_state=0,test_size=0.3)
# print(X_test)
# print(X_train)


#Definicion de funcion para realizar KNN
def Knnp(lista):
    #Lectura de documentos CSV de entrenamiento
    signo_ok = pd.read_csv("Click_ok.csv")
    signo_ok_2 = pd.read_csv("Click_ok2.csv")
    signo_ok_3 = pd.read_csv("Click_ok3.csv")

    #Longitud de las filas de las caracteristicas
    dis1 = len(signo_ok)
    dis2 = len(signo_ok_2)
    dis3 = len(signo_ok_3)

    #Se halla el CSV con menor longitud en las filas
    dis_t= [dis1,dis2,dis3]
    dis_min= min(dis_t)
    index_min= dis_t.index(dis_min)

    #Recorte de las caracteristicas segun el CSV con menor longitud en las filas
    if index_min==0:
        signo_ok_2 = signo_ok_2.drop(range(dis1, dis2), axis=0)
        signo_ok_3 = signo_ok_3.drop(range(dis1, dis3), axis=0)
    elif(index_min==1):
        signo_ok = signo_ok.drop(range(dis2, dis1), axis=0)
        signo_ok_3 = signo_ok_3.drop(range(dis2, dis3), axis=0)
    else:
        signo_ok= signo_ok.drop(range(dis3, dis1), axis=0)
        signo_ok_2= signo_ok_2.drop(range(dis3, dis2), axis=0)


    #Concatenacion de los CSV de entrenamiento
    sig_t= pd.concat([signo_ok,signo_ok_2,signo_ok_3],axis=0)

    #Se escoge los X y Y, de entrenamiento y testeo
    X_train, X_test, y_train, y_test = train_test_split(sig_t.iloc[:, :-1], sig_t.iloc[:, -1], random_state=0,test_size=0.60)

    ## Normalizacion de datos
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ## Entrenamiento de Knn
    knn = KNeighborsClassifier(n_neighbors = 1,weights='distance',metric='chebyshev', metric_params=None,algorithm='brute')
    knn.fit(X_train, y_train)
    #Transformacion de la lista de datos nuevos ingresados
    arreglo = np.array(lista)
    a = scaler.transform(arreglo.reshape(1, arreglo.shape[0]))
    #Prediccion de Knn con los nuevos datos
    y_predict = knn.predict(a)
    return y_predict



