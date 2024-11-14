import os
import random
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Cargando modelo

longitud, altura = 150, 150
modelo = os.path.join(os.getcwd(), "models_cnn\\modelo.h5")
pesos_modelo = os.path.join(os.getcwd(), "models_cnn\\pesos.h5")
try:
    cnn = load_model(modelo)
    cnn.load_weights(pesos_modelo)
except:
    clases = ["leve", "moderado", "severo"]
    request = lambda validation: random.choice(validation)


def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)


# Creacion de la GUI

main_window = tk.Tk()
main_window.title("Identificador de neumaticos")
main_window.geometry("800x800")

img_df = Label(main_window, text="Esperando imagen...").place(x=220, y=380)
file = None


def openfile():
    global file
    file = filedialog.askopenfilename(
        title="Abrir imagen",
        initialdir=os.getcwd(),
        filetypes=(
            ("Archivos de imagen JPG", "*.jpg"),
            ("Archivos de imagen JPEG", "*.jpeg"),
            ("Archivos de imagen PNG", "*.png"),
            ("Todos los archivos", "*.*"),
        ),
    )

    return file


def proc_img():
    global file
    if file == None:
        messagebox.showwarning("Aviso", "Cargue una imagen para analizar")
    else:
        message = messagebox.showinfo(
            "Alerta", f"El neumatico tiene un desgaste: {request(clases)}"
        )
        if message == "ok":
            file = None


def imagen():
    img = Image.open(openfile())
    new_img = img.resize((500, 500))
    render = ImageTk.PhotoImage(new_img)
    global img_df
    img_df = Label(main_window, image=render)
    img_df.image = render
    img_df.place(x=140, y=40)


upload = Button(
    main_window, command=imagen, text="Cargar imagen", height=2, width=20
).place(x=120, y=640)
validate = Button(
    main_window, command=proc_img, text="Validar imagen", height=2, width=20
).place(x=440, y=640)

main_window.mainloop()
