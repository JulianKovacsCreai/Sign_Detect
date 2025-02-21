import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Parámetros
img_size = 224  # Tamaño de las imágenes para el modelo
batch_size = 32
num_classes = 5  # Vocales: A, E, I, O, U

# Cargar dataset con aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizar imágenes
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # Ajustar píxeles alrededor del borde
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Solo normalización en validación

train_generator = train_datagen.flow_from_directory(
    'C:/Users/Julian.Drago/JULIAN/Creai_Investigacion/LenguajeVocales/dataset/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'C:/Users/Julian.Drago/JULIAN/Creai_Investigacion/LenguajeVocales/dataset/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Usar MobileNetV2 preentrenado como base
base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

# Congelar las capas base para que no se actualicen durante el entrenamiento
base_model.trainable = True

# Congelar las primeras capas para evitar sobreajuste
for layer in base_model.layers[:100]:  
    layer.trainable = False

# Construir el modelo
model = keras.Sequential([
    base_model,  # Agregar MobileNetV2 como parte de nuestro modelo
    layers.GlobalAveragePooling2D(),  # Pooling global para reducir dimensiones
    layers.Dense(128, activation='relu'),  # Capa densa con ReLU
    layers.Dropout(0.5),  # Dropout para regularización
    layers.Dense(num_classes, activation='softmax')  # Capa de salida con softmax para clasificación múltiple
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.00001),  
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

# Mostrar resumen del modelo
model.summary()

# Entrenar el modelo
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30  # Agregar más épocas si es necesario
)

# Guardar el modelo entrenado
model.save('sign_language_vocales.h5')

# Evaluar el modelo con los datos de validación
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')
