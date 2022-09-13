#detector de codigos ArUco
import cv2 #libreria opencv para procesamiento de imagenes
import numpy as np #libreria que permite usar arreglos

#definir diccionario con posibles tipos de marcadores que se pueden detectar
aruco_dict = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}

parameters = cv2.aruco.DetectorParameters_create() #instanciar parametros para detector de marcadores aruco

cap = cv2.VideoCapture(0) #iniciar captura de video

while(True):
    ret, frame = cap.read() #extraer frame de captura de vide
    
    #iterar por diccionario que contiene posibles tipos de marcadores aruco que pueden presentarse en la imagen
    for (arucoName, arucoDict) in aruco_dict.items():
        # revisar si se detectan marcadores aruco en imagen
        arucoDict = cv2.aruco.Dictionary_get(arucoDict)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        if len(corners) > 0:
            print("[INFO] detected {} markers for '{}'".format(len(corners), arucoName))
            ids = ids.flatten()
            # iterar sobre marcadores aruco detectados
            marker_centers = [] #lista que contiene coordenadas centrales de marcadores detectados
            for (markerCorner, markerID) in zip(corners, ids):
			# extraer las esquinas de los marcadores aruco que se detectaron
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
			# convertir coordenada de maracadores aruco que se encontraron a integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
    #revisar si el usuario quiere finalizar stream
                # dibujar rectangulo alrededor de aruco detectado
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                #obtener coordenadas x,y de marcador aruco
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                marker_centers.append([cX,cY]) #agregar coordenada de centro de marcador a diccionario
            print("coordenadas centrales de marcadores detectados: ")
            print(marker_centers)
            if (marker_centers[0][0]) >= 241:
                print("Rover tiene que dirigirse a la derecha")
            elif (marker_centers[0][0] <= 239):
                print("Rover tiene que dirigirse a la izquierda")
            
    #print("img size = " + str(frame.shape)) #tamano de imagen es 480x640
    #centro de imagen es pixel 240 en x, por lo tanto ese es el setpoint
    #queremos que camara siempre apunte a dicho setpoint
    cv2.imshow('frame', frame) #mostrar frame en ventana
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release() #finalizar captura de video con camara 
cv2.destroyAllWindows()
