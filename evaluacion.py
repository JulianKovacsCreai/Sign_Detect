import cv2
import numpy as np
from tensorflow import keras
import SeguimientoManos as sm

# Inicializar captura de video
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Ancho
cap.set(4, 480)  # Alto

# Cargar el modelo entrenado
model = keras.models.load_model('sign_language_vocales.h5')

# Inicializar detector de manos
detector = sm.detectormanos(Confdeteccion=0.9)

# Clases del modelo (ajústalas si es necesario)
clases = ['A', 'E', 'I', 'O', 'U']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.encontrarmanos(frame, dibujar=True)  # Detectar manos
    lista1, bbox, mano = detector.encontrarposicion(frame, 
                                                    manoNumero=0, 
                                                    dibujarPuntos=True, 
                                                    dibujarBox=True,
                                                    color=[0,255,0]) 

    if mano == 1:  # Si hay una mano detectada
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, xmin - 40) 
        ymin = max(0, ymin - 40)
        xmax = min(frame.shape[1], xmax + 40)
        ymax = min(frame.shape[0], ymax + 40)  # Ajustar márgenes sin salir del frame

        recorte = frame[ymin:ymax, xmin:xmax]  # Recortar la imagen de la mano
        if recorte.size != 0:
            recorte = cv2.resize(recorte, (224, 224))  # Redimensionar a 224x224

            # Preprocesamiento antes de la predicción
            recorte = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
            recorte = recorte.astype('float32') / 255.0  # Normalizar valores entre 0 y 1
            recorte = np.expand_dims(recorte, axis=0)  # Agregar dimensión de batch

            # Hacer predicción con el modelo
            resultado = model.predict(recorte)
            pred_label = np.argmax(resultado)  # Obtener la clase con mayor probabilidad
            pred_vocal = clases[pred_label]  # Convertir índice a vocal
            confianza = np.max(resultado)  # Obtener confianza de la predicción

            # Mostrar la predicción en pantalla
            texto_prediccion = f'Pred: {pred_vocal} ({confianza*100:.2f}%)'
            cv2.putText(frame, texto_prediccion, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección de Lenguaje de Señas', frame)  # Mostrar el video con predicción
    if cv2.waitKey(1) & 0xFF == 27:  # Salir con la tecla ESC
        break

cap.release()
cv2.destroyAllWindows()
