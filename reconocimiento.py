#Librerias
import cv2
import mediapipe as mp
import etiquetado_datos
import numpy as np
import tensorflow as tf

# Configuración de objetos de mediapipe holistic y modelo
mpMarcas = mp.solutions.holistic
marcasCuerpo = mpMarcas.Holistic()

threshold = 0.8

# Formatos para guardar datos en arreglos
formato_pose = ['x', 'y', 'z', 'visibility']
formato_general = ['x', 'y', 'z']
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()

    num_classes = min(len(res), len(actions), len(colors))

    # Itera sobre el número de clases
    for num in range(num_classes):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(res[num] * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# Función principal para procesar el flujo de datos de la cámara
def main():
    camara = cv2.VideoCapture(1)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    sequence, sentence, predictions = [], [], []
    model = tf.keras.models.load_model('./modelo/modelo_gestos.keras')

    while True:
        exito, imagen = camara.read()
        if not exito or imagen is None:
            print("No se pudo capturar el frame, intentando nuevamente...")
            cv2.waitKey(100)
            camara = cv2.VideoCapture(1)
            continue

        imagen, evaluacion = modelo_holistic(imagen, marcasCuerpo)
        dibujo_marcas(imagen, evaluacion)
        keypoints = almacenar_arreglos(evaluacion)

        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(etiquetado_datos.gestos_array[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Visualización de probabilidades y predicciones
            if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if etiquetado_datos.gestos_array[np.argmax(res)] != sentence[-1]:
                        sentence.append(etiquetado_datos.gestos_array[np.argmax(res)])
                else:
                    sentence.append(etiquetado_datos.gestos_array[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            imagen = prob_viz(res, etiquetado_datos.gestos_array, imagen, colors)

        cv2.rectangle(imagen, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(imagen, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Imagen capturada', imagen)
        cv2.waitKey(1)


# Función que procesa la imagen con mediapipe holistic
def modelo_holistic(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    results = modelo.process(imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    return imagen, results


# Función que dibuja marcas de reconocimiento en la imagen
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


# Función para extraer puntos clave (landmarks) de las marcas obtenidas
def extraer_landmarks(marcas, dims, tamano_predeterminado, action_bool):
    if marcas is not None and action_bool:
       return np.array([[getattr(cord, dim) for dim in dims] for cord in marcas.landmark]).flatten()
    else:
        return np.zeros(tamano_predeterminado)


# Función que almacena datos de puntos clave en arreglos
def almacenar_arreglos(resultados, pose_bool = True,face_bool= True,left_hand_bool= True,right_hand_bool= True):
    pose = extraer_landmarks(resultados.pose_landmarks, formato_pose, 132, pose_bool)
    rostro = extraer_landmarks(resultados.face_landmarks, formato_general, 1404,face_bool)
    mano_derecha = extraer_landmarks(resultados.right_hand_landmarks, formato_general, 63,right_hand_bool)
    mano_izquierda = extraer_landmarks(resultados.left_hand_landmarks, formato_general, 63,left_hand_bool)
    return np.concatenate([pose, rostro, mano_derecha, mano_izquierda])


# Ejecutar main solo si el script es el principal
if __name__ == '__main__':
    main()
