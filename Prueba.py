
##Importacion de librerias
import cv2
import mediapipe as mp
import mouse
import ctypes
from datetime import datetime
from Knn_real import Knnp

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
lista=[]
X_list=[]
Y_list=[]
dedos = (dir(mp_hands.HandLandmark))[0:21]
#Definicion de parametros de metodo mp.hands.Hands
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    # Mientras la camara funcione se toman datos
    while cap.isOpened():
        #Lectura de camara
        success, image = cap.read()

        # Revision de correcto funcionamiento de camara
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        #Se refleja la imagen de manera horinzontal y conversion de formato BGR a RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        height, width,z = image.shape
        #Imagenes que no sean escribibles de pasan por referencia
        image.flags.writeable = False

        # Obtencion de puntos dados por media pipe de la mano
        results = hands.process(image)
        resultados = results.multi_hand_landmarks
        dedos = (dir(mp_hands.HandLandmark))[0:21]

        #Se dibujan las anotaciones de la mano en la imagen
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Verificacion de que haya datos por leer
        if results.multi_hand_landmarks:
            # Se recorren los resultados obtenidos con un ciclo for
            for idx, hand_landmarks in enumerate(resultados):
                if(1):
                    for b, d in enumerate(dedos):
                        # Lectura de puntos en X
                        X_point = int(hand_landmarks.landmark[getattr(
                            mp_hands.HandLandmark, d)].x * ancho)
                        lista.append(X_point) #Se anade a las listas los puntos leidos

                        # Lectura de puntos en Y
                        Y_point = int(hand_landmarks.landmark[getattr(
                            mp_hands.HandLandmark, d)].y * alto)
                        lista.append(Y_point)#Se anade a las listas los puntos leidos

                tipo=Knnp(lista) #Uso de funcion para aplicar Knn
                lista=[] #Reinicio de lista
                print(tipo)


                ##Movimiento Mouse
                x_8 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y_8 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                cv2.circle(image,(x_8,y_8),3,(255,0,0),3)
                x_mouse = int(x_8*ancho/width)
                y_mouse = int(y_8*alto/height)
                tiempo_2+=1

                if (((resultados) != None)and tiempo_2==1):
                    tiempo_2=0
                    mouse.move(x_mouse,y_mouse)

                ## Click izquierdo
                tiempo=tiempo+1

                if(tipo==2 and tiempo>10):
                    #mouse.click("right")
                    tiempo=0
                    print("Hice click izquierdo")

                ## Click derecho
                if(tipo==1 and tiempo>10):
                    #mouse.click("left")
                    tiempo=0
                    print("Hice click derecho")

        # Se muestra en pantalla lo captado por la camara en tiempo real
        cv2.imshow('MediaPipe Hands', image)
        # En caso de oprimir la tecla q se finaliza el programa
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

#Se finaliza el captado de camara
cap.release()

