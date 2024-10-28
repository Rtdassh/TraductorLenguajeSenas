import  os
import numpy as np
#Obtener cantidad total de gestos a entrenar. Almacenamiento en un arreglo
ruta_datos_entrenamiento = os.path.join('./recoleccion_datos')
carpetas_entrenamiento = [nombre for nombre in os.listdir(ruta_datos_entrenamiento)
                          if os.path.isdir(os.path.join(ruta_datos_entrenamiento, nombre))]

#Se crea el diccionario de etiquetas
diccionario_etiquetas = {label:num for num, label in enumerate(carpetas_entrenamiento)}

secuencias, etiquetas = [], []

for gesto, label in diccionario_etiquetas.items():
    directorio_gesto = os.path.join(ruta_datos_entrenamiento, gesto)
    listado_secuencias = np.array(os.listdir(directorio_gesto)).astype(int)

    for secuencia in listado_secuencias:
        window = [
            np.load(os.path.join(directorio_gesto, str(secuencia), f"{frame_num}.npy"))
            for frame_num in range(50)
        ]
        secuencias.append(window)
        etiquetas.append(label)
