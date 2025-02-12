# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:38:54 2024

@author: Gerardo
"""

import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------
import time
import random  # Para simular un ángulo en tiempo real
import serial
import sys  # Importa el módulo sys para usar sys.exit()
from time import sleep

# Inicializamos el puerto de serie a 9600 baud
try:
    ser = serial.Serial('COM4', 115200)  
except serial.SerialException as e:
    print(f"No se pudo abrir el puerto: {e}")
    sys.exit()  # Salir de forma segura si no se puede abrir el puerto

sleep(2)  # Pausa para asegurarse de que el puerto se inicializa correctamente
#---------------------------------------------------------------------
start_time = time.time()

# Inicializar el gráfico
plt.figure()  # Crear una nueva figura
plt.xlim(0, 640)  # Limites en el eje x
plt.ylim(0, 480)  # Limites en el eje y
plt.grid(True)  # Activar la cuadrícula
plt.ion()  # Activar modo interactivo

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
   
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        #print('hand landmarks',results.multi_hand_landmarks)
        # Imprimir el tamaño del display
      
        
        if results.multi_hand_landmarks is not None:
           
            for hand_landmarks in results.multi_hand_landmarks:
               
                x4=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                y4=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
             
                x8=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y8=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
             
             
                x12=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
                y12=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
                
                x16=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
                y16=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
                
                x20=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
                y20=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)
                
                x0=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                y0=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
                z0=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z * 400)
                
                #-------------------------------------------------------------------------------
                
                x3=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * width)
                y3=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * height)
             
                x7=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * width)
                y7=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * height)
             
                x11=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * width)
                y11=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * height)
                
                x15=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * width)
                y15=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * height)
                
                x19=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * width)
                y19=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * height)
                
                #--------------------------------------------------------------------------------------
            
                x2=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width)
                y2=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * height)
                z2=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z *400)
            
                x6=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * width)
                y6=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * height)
            
                x10=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * width)
                y10=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * height)
               
                x14=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * width)
                y14=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * height)
               
                x18=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * width)
                y18=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * height)
                
                
                #--------------------------------------------------------------------------------------
            
                x1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * width)
                y1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * height)
            
                x5=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width)
                y5=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height)
            
                x9=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width)
                y9=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height)
               
                x13=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * width)
                y13=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * height)
               
                x17=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width)
                y17=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height)
                z17=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z *400)
                
                x22=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width-100)
                z22=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z* 400)
              
                
               
                
                cv2.circle(frame,(x4,y4),3,(0,0,255),4)
                cv2.circle(frame,(x8,y8),3,(0,0,255),4)
                cv2.circle(frame,(x12,y12),3,(0,0,255),4)
                cv2.circle(frame,(x16,y16),3,(0,0,255),4)
                cv2.circle(frame,(x20,y20),3,(0,0,255),4)
                cv2.circle(frame,(x0,y0),3,(0,0,255),4)
                
                #cv2.circle(frame,(x3,y3),3,(0,0,255),4)
                #cv2.circle(frame,(x7,y7),3,(0,0,255),4)
                #cv2.circle(frame,(x11,y11),3,(0,0,255),4)
                #cv2.circle(frame,(x15,y15),3,(0,0,255),4)
                #cv2.circle(frame,(x19,y19),3,(0,0,255),4)
                
                cv2.circle(frame,(x2,y2),3,(0,0,255),4)
                #cv2.circle(frame,(x6,y6),3,(0,0,255),4)
                #cv2.circle(frame,(x10,y10),3,(0,0,255),4)
                #cv2.circle(frame,(x14,y14),3,(0,0,255),4)
                #cv2.circle(frame,(x18,y18),3,(0,0,255),4)
                
                #cv2.circle(frame,(x1,y1),3,(0,0,255),4)
                cv2.circle(frame,(x5,y5),3,(0,0,255),4)
                cv2.circle(frame,(x9,y9),3,(0,0,255),4)
                cv2.circle(frame,(x13,y13),3,(0,0,255),4)
                cv2.circle(frame,(x17,y17),3,(0,0,255),4)
                
                
                # cv2.line(frame, [x4, y4], [x3, y3], (255, 255, 0), thickness=2)
                # cv2.line(frame, [x8, y8], [x7, y7], (255, 255, 0), thickness=2)
                # cv2.line(frame, [x11, y11], [x12, y12], (255, 255, 0), thickness=2)
                # cv2.line(frame, [x15, y15], [x16, y16], (255, 255, 0), thickness=2)
                # cv2.line(frame, [x19, y19], [x20, y20], (255, 255, 0), thickness=2)
               
                
                
                plt.clf()  # Limpiar la figura
                plt.xlim(0, 640)  # Limites en el eje x
                plt.ylim(480,0)  # Limites en el eje y
                plt.grid(True)  # Activar la cuadrícula

  
                # Dibujar los puntos
                plt.plot(x4, y4, 'co', markerfacecolor='c', markersize=7)  
                plt.plot(x8, y8, 'bo', markerfacecolor='b', markersize=7)  
                plt.plot(x12, y12, 'yo', markerfacecolor='y', markersize=7)  
                plt.plot(x16, y16, 'go', markerfacecolor='g', markersize=7)  
                plt.plot(x20, y20, 'mo', markerfacecolor='m', markersize=7)  
                plt.plot(x0, y0, 'black', markerfacecolor='black', markersize=7)  
               
 
                plt.plot(x3, y3, 'co', markerfacecolor='c', markersize=7)  
                plt.plot(x7, y7, 'bo', markerfacecolor='b', markersize=7) 
                plt.plot(x11, y11, 'yo', markerfacecolor='y', markersize=7)  
                plt.plot(x15, y15, 'go', markerfacecolor='g', markersize=7)  
                plt.plot(x19, y19, 'mo', markerfacecolor='m', markersize=7) 
               
               
                plt.plot(x2, y2, 'co', markerfacecolor='c', markersize=7)  
                plt.plot(x6, y6, 'bo', markerfacecolor='b', markersize=7) 
                plt.plot(x10, y10, 'yo', markerfacecolor='y', markersize=7)  
                plt.plot(x14, y14, 'go', markerfacecolor='g', markersize=7)  
                plt.plot(x18, y18, 'mo', markerfacecolor='m', markersize=7)  
               
                plt.plot(x1, y1, 'black', markerfacecolor='black', markersize=7)  
                plt.plot(x5, y5, 'bo', markerfacecolor='b', markersize=7) 
                plt.plot(x9, y9, 'yo', markerfacecolor='y', markersize=7)  
                plt.plot(x13, y13, 'go', markerfacecolor='g', markersize=7)  
                plt.plot(x17, y17, 'mo', markerfacecolor='m', markersize=7)  
              
               
          #--------------------------------------------------------------------------------------------------    
                plt.plot([x4, x3], [y4, y3], color='aqua', linestyle='-', linewidth=6) 
                plt.plot([x8, x7], [y8, y7], color='b', linestyle='-', linewidth=6) 
                plt.plot([x12, x11], [y12, y11], color='y', linestyle='-', linewidth=6) 
                plt.plot([x16, x15], [y16, y15], color='green', linestyle='-', linewidth=6) 
                plt.plot([x20, x19], [y20, y19], color='m', linestyle='-', linewidth=6) 
               
                plt.plot([x3, x2], [y3, y2], color='aqua', linestyle='-', linewidth=7) 
                plt.plot([x7, x6], [y7, y6], color='b', linestyle='-', linewidth=7) 
                plt.plot([x11, x10], [y11, y10], color='y', linestyle='-', linewidth=7) 
                plt.plot([x15, x14], [y15, y14], color='green', linestyle='-', linewidth=7) 
                plt.plot([x19, x18], [y19, y18], color='m', linestyle='-', linewidth=7) 
               
                plt.plot([x2, x1], [y2, y1], color='black', linestyle='-', linewidth=7) 
                plt.plot([x6, x5], [y6, y5], color='b', linestyle='-', linewidth=7) 
                plt.plot([x10, x9], [y10, y9], color='y', linestyle='-', linewidth=7) 
                plt.plot([x14, x13], [y14, y13], color='green', linestyle='-', linewidth=7) 
                plt.plot([x18, x17], [y18, y17], color='m', linestyle='-', linewidth=7) 
               
                plt.plot([x17, x13], [y17, y13], color='black', linestyle='-', linewidth=7) 
                plt.plot([x13, x9], [y13, y9], color='black', linestyle='-', linewidth=7) 
                plt.plot([x9, x5], [y9, y5], color='black', linestyle='-', linewidth=7) 
                plt.plot([x5, x2], [y5, y2], color='black', linestyle='-', linewidth=7) 
                plt.plot([x0, x1], [y0, y1], color='black', linestyle='-', linewidth=7)
                plt.plot([x0, x1], [y0, y1], color='black', linestyle='-', linewidth=7) 
                plt.plot([x0, x17], [y0, y17], color='black', linestyle='-', linewidth=7)
               
                #-------------------------------------------------------------------
                plt.plot([x8, x5], [y8, y5], color='r', linestyle=':', linewidth=2) 
                plt.plot([x12, x9], [y12, y9], color='r', linestyle=':', linewidth=2) 
                plt.plot([x16, x13], [y16, y13], color='r', linestyle=':', linewidth=2) 
                plt.plot([x17, x20], [y17, y20], color='r', linestyle=':', linewidth=2) 
                plt.plot([x4, x2], [y4, y2], color='r', linestyle=':', linewidth=2) 
                #----------------pulgar--------------------------------------------------------
                landmark_4 = np.array([x4, y4, ])  
                landmark_2 = np.array([x2, y2, ]) 
                landmark_1 = np.array([x17, y17, ])  

                # Vectores entre los puntos
                vector2 = landmark_4 - landmark_2  # Vector de Landmark 2 a Landmark 3
                vector1 = landmark_2 - landmark_1  # Vector de Landmark 4 a Landmark 3

                # Calcular el ángulo entre los vectores
                cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector2) * np.linalg.norm(vector1))
                angle = np.arccos(cosine_angle)  # Resultado en radianes

                # Convertir a grados (opcional)
                angle1 = np.degrees(angle)
                
                
                #------------------indice---------------------------------------------------------      
                landmark_8 = np.array([x8, y8, ])  
                landmark_6 = np.array([x6, y6, ]) 
                landmark_5 = np.array([x5, y5, ])  

                # Vectores entre los puntos
                vector3 = landmark_5 - landmark_6  # Vector de Landmark 2 a Landmark 3
                vector4 = landmark_6 - landmark_8  # Vector de Landmark 4 a Landmark 3

                # Calcular el ángulo entre los vectores
                cosine_angle = np.dot(vector3, vector4) / (np.linalg.norm(vector3) * np.linalg.norm(vector4))
                angle2 = np.arccos(cosine_angle)
                
                # Convertir a grados (opcional)
                angle2 =180-np.degrees(angle2)
       
                #-------------------medio-----------------------------------------------------    
                landmark_12 = np.array([x12, y12])  
                landmark_10 = np.array([x10, y10]) 
                landmark_9 = np.array([x9, y9])  

                # Vectores entre los puntos
                vector6 = landmark_12 - landmark_10  # Vector de Landmark 2 a Landmark 3
                vector5 = landmark_10 - landmark_9  # Vector de Landmark 4 a Landmark 3

                # Calcular el ángulo entre los vectores
                cosine_angle = np.dot(vector5, vector6) / (np.linalg.norm(vector5) * np.linalg.norm(vector6))
                angle3 = np.arccos(cosine_angle)  # Resultado en radianes

                # Convertir a grados (opcional)
                angle3 = 180-np.degrees(angle3)
               
                #-----------------------anular-------------------------------------------------   
                landmark_16 = np.array([x16, y16])  
                landmark_14 = np.array([x14, y14]) 
                landmark_13 = np.array([x13, y13])  

                # Vectores entre los puntos
                vector8 = landmark_16 - landmark_14  # Vector de Landmark 2 a Landmark 3
                vector7 = landmark_14 - landmark_13  # Vector de Landmark 4 a Landmark 3

                # Calcular el ángulo entre los vectores
                cosine_angle = np.dot(vector7, vector8) / (np.linalg.norm(vector7) * np.linalg.norm(vector8))
                angle4 = np.arccos(cosine_angle)  # Resultado en radianes

                # Convertir a grados (opcional)
                angle4 = 180-np.degrees(angle4)                  
                
                #-----------------------meñique-------------------------------------------------   
                landmark_20 = np.array([x20, y20])  
                landmark_18 = np.array([x18, y18]) 
                landmark_17 = np.array([x17, y17])  

                # Vectores entre los puntos
                vector10 = landmark_20 - landmark_18  # Vector de Landmark 2 a Landmark 3
                vector9 = landmark_18 - landmark_17  # Vector de Landmark 4 a Landmark 3

                # Calcular el ángulo entre los vectores
                cosine_angle = np.dot(vector9, vector10) / (np.linalg.norm(vector9) * np.linalg.norm(vector10))
                angle5 = np.arccos(cosine_angle)  # Resultado en radianes

                # Convertir a grados (opcional)
                angle5 = 180-np.degrees(angle5)
                #print(f'{angle1:.2f}\t {angle2:.2f}\t {angle3:.2f}\t{angle4:.2f}\t{angle5:.2f}')                  
               
                # plt.pause(0.02)  # Espera medio segundo entre actualizaciones
                
                # ----------------------------------------Calcular el ángulo la base
                landmark_1717= np.array([x17, z17])  
                landmark_22 = np.array([x2, z2]) 
                landmark_2222 = np.array([x22, z22])  

                # Vectores entre los puntos
                vectorz1 = landmark_1717 - landmark_22  # Vector de Landmark 2 a Landmark 3
                vectorz2 = landmark_22 - landmark_2222  # Vector de Landmark 4 a Landmark 3

                # Calcular el ángulo entre los vectores
                cosine_angle = np.dot(vectorz1, vectorz2) / (np.linalg.norm(vectorz1) * np.linalg.norm(vectorz2))
                angle6= np.arccos(cosine_angle)  # Resultado en radianes

                # Convertir a grados (opcional)
                angle6= np.degrees(angle6)                     
                
                end_time = time.time()
                angulos_str = f"{angle1:.0f},{angle2:.0f},{angle3:.0f},{angle4:.0f},{angle5:.0f},{angle6:.0f}\n"
                
                
                
                print(f'{angle1:.0f}\t{angle2:.0f}\t{angle3:.0f}\t{angle4:.0f}\t{angle5:.0f}\t{angle6:.0f}\t{end_time - start_time:.2f}')
              
                #sleep(.02)

               
                
            plt.ioff()  # Desactivar modo interactivo
            plt.show()  # Mantener la ventana abierta al final
            ser.write(angulos_str.encode())    
             
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
ser.close()  # Cierra la conexión serial
cap.release()
cv2.destroyAllWindows()



  
