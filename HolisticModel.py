#Librerias
import cv2
import mediapipe as mp
import numpy as np
import os

#Objetos de mediapipe holistic
mpMarcas = mp.solutions.holistic
marcasCuerpo = mpMarcas.Holistic()
#Formatos para guardar datos en arreglos
formato_pose = ['x', 'y', 'z', 'visibility']
formato_general = ['x', 'y', 'z']


# Funcion principal donde se almacena el flujo de programa inicial
def main():
    # Instancia camara y fijacion de tamaño
    camara = cv2.VideoCapture(1)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Lee la camara, procesa los datos y dibuja puntos en la imagen
    while True:
        exito, imagen = camara.read()
        if not exito or imagen is None:
            print("No se pudo capturar el frame, intentando nuevamente...")
            cv2.waitKey(100)
            camara = cv2.VideoCapture(1)
            continue
        imagen, evaluacion = modelo_holistic(imagen, marcasCuerpo)
        dibujo_marcas(imagen, evaluacion)
        test = almacenar_arreglos(evaluacion)
        print(test[0])

        cv2.imshow('Imagen capturada', imagen)
        cv2.waitKey(1)

#Funcion que procesa la imagen con mediapipe holistic
def modelo_holistic(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    results = modelo.process(imagen)  # Make prediction
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    return imagen, results

#Funcion que dibuja marcas de reconocimiento en su respectivo campo
def dibujo_marcas(imagen, resultados):
    mp_dibujo = mp.solutions.drawing_utils
    if resultados.left_hand_landmarks:
        mp_dibujo.draw_landmarks(imagen, resultados.left_hand_landmarks, mpMarcas.HAND_CONNECTIONS)
    if resultados.right_hand_landmarks:
        mp_dibujo.draw_landmarks(imagen, resultados.right_hand_landmarks, mpMarcas.HAND_CONNECTIONS)
    if resultados.pose_landmarks:
        mp_dibujo.draw_landmarks(imagen, resultados.pose_landmarks, mpMarcas.POSE_CONNECTIONS)
    if resultados.face_landmarks:
        mp_dibujo.draw_landmarks(imagen, resultados.face_landmarks, mpMarcas.FACEMESH_CONTOURS)

#Funcion base para guardar datos en arreglos
def extraer_landmarks(marcas, dims, tamano_predeterminado):
    return (np.array([[getattr(cord, dim) for dim in dims] for cord in marcas.landmark]).flatten()
            if marcas else np.zeros(tamano_predeterminado))

#Funcion que guarda datos en arreglos
def almacenar_arreglos(resultados):
    pose = extraer_landmarks(resultados.pose_landmarks, formato_pose, 132)
    rostro = extraer_landmarks(resultados.face_landmarks, formato_general, 1404)
    mano_derecha = extraer_landmarks(resultados.right_hand_landmarks, formato_general, 63)
    mano_izquierda = extraer_landmarks(resultados.left_hand_landmarks, formato_general, 63)
    return np.concatenate([pose, rostro, mano_derecha, mano_izquierda])

#main ejecutara unicamente si se encuentra en este modulo
if __name__ == '__main__':
    main()