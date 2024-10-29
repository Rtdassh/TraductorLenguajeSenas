# 🌐 Language IA (LIA)

## 📄 Resumen
**LIA** es una aplicación web diseñada para facilitar la comunicación entre personas con discapacidad del habla o auditiva y aquellas que no conocen el lenguaje de señas.

## 🎯 Objetivo
Desarrollar una herramienta accesible que permita la comunicación efectiva entre personas con discapacidades del habla/auditiva y sus interlocutores, eliminando barreras de comunicación y promoviendo la inclusión.

## 📊 Contexto
Según la Encuesta Nacional sobre Discapacidad (ENDIS) de 2016, el 10.2% de la población posee alguna discapacidad, aunque las estadísticas específicas para personas con discapacidad auditiva o del habla son limitadas. Este proyecto busca atender las necesidades de comunicación de este grupo, que enfrenta dificultades para integrarse plenamente en la sociedad debido a barreras lingüísticas.

La Universidad Rafael Landívar actualmente cuenta con un Programa de Apoyo Académico que ofrece tutorías y apoyo personalizado a estudiantes con discapacidades, pero aún no dispone de un programa de lengua de señas. **LIA** surge como una propuesta para mejorar la accesibilidad y fomentar la inclusión.

## 🛠️ Descripción General del Programa
**LIA** permite traducir en tiempo real gestos de lenguaje de señas a texto, facilitando la comunicación. Esta funcionalidad se logra mediante el uso de tecnologías avanzadas de procesamiento de imágenes y redes neuronales.

## 🔍 Estructura de LIA

1. **🎥 Captura y Procesamiento de Imágenes**
   - Librerías utilizadas: **MediaPipe** y **OpenCV**.
   - **MediaPipe** identifica puntos clave de las manos en tiempo real, mientras que **OpenCV** facilita la captura de video y preprocesamiento de imágenes para extraer estos puntos.

2. **🔧 Preprocesamiento de Datos**
   - Los puntos clave capturados se normalizan y organizan en secuencias temporales para representar los movimientos del lenguaje de señas.
   - Herramientas: **NumPy**.

3. **🧠 Creación y Entrenamiento de la Red Neuronal Recurrente (RNN)**
   - El modelo de RNN/LSTM se entrena usando **TensorFlow** y **Keras**, permitiendo que la IA asocie patrones de gestos con palabras o frases específicas.

4. **✅ Validación del Modelo**
   - Para evaluar y ajustar el modelo, se utilizan **Scikit-learn** y **Matplotlib**, proporcionando métricas de precisión y visualización de los resultados.

5. **🌐 Integración en la Aplicación Web**
   - La cámara del usuario captura gestos continuamente, que luego son procesados en un servidor y traducidos en tiempo real a texto en la interfaz de usuario.

## 💻 Tecnologías Utilizadas
- **Front-End**: HTML, JavaScript y CSS.
- **Procesamiento de Imágenes y IA**: MediaPipe, OpenCV, NumPy, TensorFlow, Keras, Scikit-learn, Matplotlib.

## 🚀 Despliegue
La aplicación LIA está desplegada en una página web que utiliza **HTML, JavaScript y CSS** para la interfaz, asegurando una experiencia interactiva y accesible para los usuarios.

## 🤝 Contribución
Las contribuciones al desarrollo de LIA son bienvenidas. Sugerencias, mejoras y reportes de errores pueden realizarse a través de GitHub.

---

