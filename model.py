import sys
import os
import random
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K


K.clear_session()


data_entrenamiento = os.path.join(os.getcwd(), "data\\train")
data_validacion = os.path.join(os.getcwd(), "data\\validation")


# Estableciendo Parametros

epocas = 20
longitud, altura = 150, 150
batch_size = 32
pasos = 1000
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
lr = 0.0004


# Cargando imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode="categorical",
)

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode="categorical",
)

validacion = ["leve", "moderado", "severo"]
request = lambda validation: random.choice(validation)

cnn = Sequential()
cnn.add(
    Convolution2D(
        filtrosConv1,
        tamano_filtro1,
        padding="same",
        input_shape=(longitud, altura, 3),
        activation="relu",
    )
)
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation="softmax"))

# Compilando modelo

cnn.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(lr=lr),
    metrics=["accuracy"],
)


cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps,
)

target_dir = os.path.join(os.getcwd(), "models_cnn")
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save(os.path.join(os.getcwd(), "models_cnn\\modelo.h5"))
cnn.save_weights(os.path.join(os.getcwd(), "models_cnn\\pesos.h5"))
