import os
import numpy as np
import cv2
import reconocimiento

DIRECTORIO_DATOS = os.path.join('./recoleccion_datos')
def main():
    creacion_directorio(DIRECTORIO_DATOS)

    #Aquí puedes ingresar el nombre del gesto que desees entrenar
    ingresar_gesto('ansioso',50,60)

def creacion_directorio(nombre_directorio):
    if not os.path.exists(nombre_directorio):
        os.makedirs(nombre_directorio)

def ingresar_gesto(nombre_signo, frames_muestra, cantidad_muestra):
    #Aquí debes colocar el índice del dispositivo que capturará la imagen
    #Suele ser 0 si solo hay una cámara conectada
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not os.path.exists(os.path.join(DIRECTORIO_DATOS, str(nombre_signo))):
        os.makedirs(os.path.join(DIRECTORIO_DATOS, str(nombre_signo)))

    while True:
        exito, imagen = cap.read()
        cv2.putText(imagen, 'Para comenzar presiona la letra s', (100, 50), cv2.FONT_ITALIC, 1, (150, 255, 44), 3,
                    cv2.LINE_AA)
        cv2.imshow('Recoleccion de datos', imagen)
        if cv2.waitKey(15) == ord('s'):
            cv2.destroyAllWindows()
            break

    contador = 0
    while contador < cantidad_muestra:
        # Crear directorio para cada conjunto de 50 frames
        carpeta_path = os.path.join(DIRECTORIO_DATOS, str(nombre_signo), str(contador))
        creacion_directorio(carpeta_path)

        for iteracion_frame in range(frames_muestra):
            exito, frame = cap.read()
            frame, evaluacion = reconocimiento.modelo_holistic(frame, reconocimiento.marcasCuerpo)
            reconocimiento.dibujo_marcas(frame, evaluacion)
            cv2.putText(frame, f'Recolectando datos para {nombre_signo} - Set {contador}, Frame {iteracion_frame + 1}',(15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            #Ajustar el valor si se desean todos los parametros en almacenar arreglos
            marcas = reconocimiento.almacenar_arreglos(evaluacion, True, True, True, True)
            directorio_numpy = os.path.join(carpeta_path, str(iteracion_frame))
            np.save(directorio_numpy, marcas)

            if iteracion_frame+1== frames_muestra:
                cv2.putText(frame, 'Set guardado, comenzara el siguiente', (100, 200), cv2.FONT_ITALIC, 0.7,(150, 255, 44),2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(20)

        contador += 1
        cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()