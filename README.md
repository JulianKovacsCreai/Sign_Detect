# Sign_Detect

## Descripción

Sign_Detect es un sistema de reconocimiento de lenguaje de señas basado en visión por computadora y aprendizaje profundo. Utiliza un modelo de redes neuronales convolucionales (CNN) para detectar y clasificar señas de la Lengua de Señas Americana en tiempo real.

## Características

- Utiliza MobileNetV2 como modelo base para la clasificación de señas.
- Entrenamiento con un dataset de imágenes de vocales (A, E, I, O, U).
- Implementación de detección en tiempo real con OpenCV y Mediapipe.
- Aumento de datos para mejorar la generalización del modelo.

## Instalación

### Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install tensorflow opencv-python mediapipe numpy
```

# Uso

## Preparación del dataset

Para preparar el dataset se ejecuta:

```bash
python Data.py
```

Se iniciará la cámara que se encargará de detectar tu mano utilizando el script de SeguimientoManos.py para tomar fotos de las diferentes señas que se guardarán en la carpeta asignada a la letra dentro de la carpeta data y así ir rellenando el dataset, donde se deberá realizar la corresponiente limpieza y ajuste de clases para el conjunto de entrenamiento y validación en la carpeta dataset.

## Entrenamiento del modelo

Para entrenar el modelo con el dataset disponible, ejecuta:

```bash
python modelo.py
```

Esto entrenará un modelo basado en MobileNetV2 y lo guardará como sign_language_vocales.h5.

## Evaluación del modelo

Para evaluar el modelo en tiempo real:

```bash
python evalucion.py
```

# Estructura del Proyecto

```bash
Sign_Detect/
├── dataset/                         # Carpeta con el dataset de entrenamiento y validación
├── Data.py                          # Script para agregar imagenes al dataset del modelo
├── SeguimientoManos.py              # Script para implementar el seguimiento de manos con mediapipe 
├── modelo.py                        # Script para entrenar el modelo
├── sign_language_vocales.h5         # Modelo desarrollado en modelo.py
├── evaluacion.py                    # Detección en tiempo real
├── README.md                        # Documentación del proyecto
```

# Mejoras Futuras

- Integración de más clases (más letras y palabras completas).
- Optimización de la detección de movimientos complejos.
- Implementación de una interfaz gráfica para facilidad de uso.
