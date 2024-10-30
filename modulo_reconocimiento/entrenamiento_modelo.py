import tensorflow as tf
import os
import etiquetado_datos

# Directorio de logs para TensorBoard
log_dir = os.path.join('Logs')
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

#Clase que termina el entrenamiento si el modelo tiene una muy buena precisión
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('categorical_accuracy') >= 0.95:
            print(f"\nSe alcanzó una precisión de {logs.get('categorical_accuracy'):.4f}, deteniendo el entrenamiento...")
            self.model.stop_training = True

stop_at_accuracy = StopAtAccuracy()

# Evita sobreentrenamiento
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Construcción del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, activation='relu', input_shape=(50, 1662)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(256, return_sequences=True, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(len(etiquetado_datos.gestos_array), activation='softmax')  # 3 clases: 'a', 'b', 'c'
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenamiento del modelo
history = model.fit(
    etiquetado_datos.X_train, etiquetado_datos.y_train,
    epochs=300,
    validation_split=0.2,
    callbacks=[tb_callback, early_stopping, stop_at_accuracy]
)

# Guardado Modelo
model.summary()
model.save('./modelo/modelo_gestos.h5')
model.save('./modelo/modelo_gestos.keras')
