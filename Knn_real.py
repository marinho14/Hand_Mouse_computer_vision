##Importacion de librerias
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#Definicion de funcion para realizar KNN
def Knnp(lista):
    # #Lectura de documentos CSV de entrenamiento
    # signo_ok = pd.read_csv("Click_ok.csv")
    # signo_ok_2 = pd.read_csv("Click_ok2.csv")
    # signo_ok_3 = pd.read_csv("Click_ok3.csv")
    #
    # #Longitud de las filas de las caracteristicas
    # dis1 = len(signo_ok)
    # dis2 = len(signo_ok_2)
    # dis3 = len(signo_ok_3)
    #
    # #Se halla el CSV con menor longitud en las filas
    # dis_t= [dis1,dis2,dis3]
    # dis_min= min(dis_t)
    # index_min= dis_t.index(dis_min)
    #
    # #Recorte de las caracteristicas segun el CSV con menor longitud en las filas
    # if index_min==0:
    #     signo_ok_2 = signo_ok_2.drop(range(dis1, dis2), axis=0)
    #     signo_ok_3 = signo_ok_3.drop(range(dis1, dis3), axis=0)
    # elif(index_min==1):
    #     signo_ok = signo_ok.drop(range(dis2, dis1), axis=0)
    #     signo_ok_3 = signo_ok_3.drop(range(dis2, dis3), axis=0)
    # else:
    #     signo_ok= signo_ok.drop(range(dis3, dis1), axis=0)
    #     signo_ok_2= signo_ok_2.drop(range(dis3, dis2), axis=0)
    #
    #
    # #Concatenacion de los CSV de entrenamiento
    # sig_t= pd.concat([signo_ok,signo_ok_2,signo_ok_3],axis=0)
    #
    # #Se escoge los X y Y, de entrenamiento y testeo
    # X_train, X_test, y_train, y_test = train_test_split(sig_t.iloc[:, :-1], sig_t.iloc[:, -1], random_state=0,test_size=0.60)

    ## lectura de entrenamiento de mano sin girar
    X_train = np.load('sample_ref3.npy')
    y_train = np.load('sample2_ref3.npy')

    ## lectura de entrenamiento de mano girada
    X_train_2 = np.load('sample_ref.npy')
    y_train_2 = np.load('sample2_ref.npy')

    X_train = np.concatenate((X_train, X_train_2), axis=0)
    y_train = np.concatenate((y_train, y_train_2), axis=0)

    X_test= np.load('sample3_ref2.npy')


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



