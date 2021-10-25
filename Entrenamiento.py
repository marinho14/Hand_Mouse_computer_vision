#Importacion de librerias
import cv2
import mediapipe as mp
import ctypes
from datetime import datetime
import pandas as pd


#Adquisicion de ancho y alto de la pantalla
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
ancho, alto = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
print(ancho, alto)

# Definicion de metodo utilizado de la libreria media pipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# Seleccion de entrada de camara
cap = cv2.VideoCapture(0)

#Variables auxiliares
i=0
tiempo=0
tiempo_2=0
#Toma de tiempo para entrenamiento
tiempo_actual= datetime.now()
tiempo_futuro=datetime.now()

#Creacion de dataframe en panda con las columnas de los puntos que seran tomados
dataframe= pd.DataFrame()
dataframe= dataframe.assign(X_point_0=None)
dataframe= dataframe.assign(Y_point_0=None)
dataframe = dataframe.assign(X_point_1=None)
dataframe = dataframe.assign(Y_point_1=None)
dataframe = dataframe.assign(X_point_2=None)
dataframe = dataframe.assign(Y_point_2=None)
dataframe = dataframe.assign(X_point_3=None)
dataframe = dataframe.assign(Y_point_3=None)
dataframe = dataframe.assign(X_point_4=None)
dataframe = dataframe.assign(Y_point_4=None)
dataframe = dataframe.assign(X_point_5=None)
dataframe = dataframe.assign(Y_point_5=None)
dataframe = dataframe.assign(X_point_6=None)
dataframe = dataframe.assign(Y_point_6=None)
dataframe = dataframe.assign(X_point_7=None)
dataframe = dataframe.assign(Y_point_7=None)
dataframe = dataframe.assign(X_point_8=None)
dataframe = dataframe.assign(Y_point_8=None)
dataframe = dataframe.assign(X_point_9=None)
dataframe = dataframe.assign(Y_point_9=None)
dataframe = dataframe.assign(X_point_10=None)
dataframe = dataframe.assign(Y_point_10=None)
dataframe = dataframe.assign(X_point_11=None)
dataframe = dataframe.assign(Y_point_11=None)
dataframe = dataframe.assign(X_point_12=None)
dataframe = dataframe.assign(Y_point_12=None)
dataframe = dataframe.assign(X_point_13=None)
dataframe = dataframe.assign(Y_point_13=None)
dataframe = dataframe.assign(X_point_14=None)
dataframe = dataframe.assign(Y_point_14=None)
dataframe = dataframe.assign(X_point_15=None)
dataframe = dataframe.assign(Y_point_15=None)
dataframe = dataframe.assign(X_point_16=None)
dataframe = dataframe.assign(Y_point_16=None)
dataframe = dataframe.assign(X_point_17=None)
dataframe = dataframe.assign(Y_point_17=None)
dataframe = dataframe.assign(X_point_18=None)
dataframe = dataframe.assign(Y_point_18=None)
dataframe = dataframe.assign(X_point_19=None)
dataframe = dataframe.assign(Y_point_19=None)
dataframe = dataframe.assign(X_point_20=None)
dataframe = dataframe.assign(Y_point_20=None)
dataframe = dataframe.assign(Tipo=None)


cont=0
X_list=[]
Y_list=[]
dedos = (dir(mp_hands.HandLandmark))[0:20]

#Definicion de parametros de metodo mp.hands.Hands
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

#Mientras la camara funcione se toman datos por un minuto
    while (cap.isOpened() and (tiempo_actual.minute+1 != tiempo_futuro.minute)):
        # Lectura de camara
        success, image = cap.read()
        #Actualizacion de tiempo para contar un minuto de entrenamiento
        tiempo_futuro = datetime.now()

        #Revision de correcto funcionamiento de camara
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue


        #Se refleja la imagen de manera horinzontal y conversion de formato BGR a RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        #Imagenes que no sean escribibles de pasan por referencia
        image.flags.writeable = False

        #Obtencion de puntos dados por media pipe de la mano
        results = hands.process(image)
        resultados = results.multi_hand_landmarks
        dedos = (dir(mp_hands.HandLandmark))[0:21]

        #Se dibujan las anotaciones de la mano en la imagen
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        #Verificacion de que haya datos por leer
        if results.multi_hand_landmarks:
            #Se recorren los resultados obtenidos con un ciclo for
            for idx, hand_landmarks in enumerate(resultados):
             if(cont==6): #Contador para no tomar demasiados datos
                for b, d in enumerate(dedos):

                    #Lectura de puntos en X
                    X_point = int(hand_landmarks.landmark[getattr(
                        mp_hands.HandLandmark, d)].x * ancho)
                    #Lectura de puntos en Y
                    Y_point = int(hand_landmarks.landmark[getattr(
                        mp_hands.HandLandmark, d)].y * alto)

                    #Se anade a las listas los puntos leidos
                    X_list.append(X_point)
                    Y_list.append(Y_point)

                    #Se anade al dataframe los datos leidos
                    dataframe = dataframe.append({'X_point_0': X_list[0], 'Y_point_0': Y_list[0],
                                                  'X_point_1': X_list[1], 'Y_point_1': Y_list[1],
                                                  'X_point_2': X_list[2], 'Y_point_2': Y_list[2],
                                                  'X_point_3': X_list[3], 'Y_point_3': Y_list[3],
                                                  'X_point_4': X_list[4], 'Y_point_4': Y_list[4],
                                                  'X_point_5': X_list[5], 'Y_point_5': Y_list[5],
                                                  'X_point_6': X_list[6], 'Y_point_6': Y_list[6],
                                                  'X_point_7': X_list[7], 'Y_point_7': Y_list[7],
                                                  'X_point_8': X_list[8], 'Y_point_8': Y_list[8],
                                                  'X_point_9': X_list[9], 'Y_point_9': Y_list[9],
                                                  'X_point_10': X_list[10], 'Y_point_10': Y_list[10],
                                                  'X_point_11': X_list[11], 'Y_point_11': Y_list[11],
                                                  'X_point_12': X_list[12], 'Y_point_12': Y_list[12],
                                                  'X_point_13': X_list[13], 'Y_point_13': Y_list[13],
                                                  'X_point_14': X_list[14], 'Y_point_14': Y_list[14],
                                                  'X_point_15': X_list[15], 'Y_point_15': Y_list[15],
                                                  'X_point_16': X_list[16], 'Y_point_16': Y_list[16],
                                                  'X_point_17': X_list[17], 'Y_point_17': Y_list[17],
                                                  'X_point_18': X_list[18], 'Y_point_18': Y_list[18],
                                                  'X_point_19': X_list[19], 'Y_point_19': Y_list[19],
                                                  'X_point_20': X_list[20], 'Y_point_20': Y_list[20], 'Tipo': 0},
                                                 ignore_index=True)
                    cont=0
                    X_list = []
                    Y_list = []
        cont+=1
        #Se muestra en pantalla lo captado por la camara en tiempo real
        cv2.imshow('MediaPipe Hands', image)

        #En caso de oprimir la tecla q se finaliza el programa
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

#Guardado de dataframe en csv
dataframe.to_csv('Click_ok2.csv', header=True, index=False)
print(dataframe)

#Se finaliza el captado de camara
cap.release()