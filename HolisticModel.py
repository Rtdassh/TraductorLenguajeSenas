import cv2
import mediapipe as mp

def dibujoMarcas():
    mpDibujo = mp.solutions.drawing_utils

def modeloHolistic():
    mpMarcas = mp.solutions.holistic
    marcasCuerpo = mpMarcas.Holistic()


def capturarImagenes():
     camara = cv2.VideoCapture(0)
     while True:
         exito, imagen = camara.read()
         colorImagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2RGB)

         cv2.imshow('Imagen capturada', imagen)
         cv2.waitKey(1)

def main():
    capturarImagenes()

if __name__ == '__main__':
    main()