# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, url_for, send_from_directory
import os
# import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFilter, ImageOps

app = Flask(__name__, static_url_path='/tmp')
# UPLOAD_FOLDER = "static/temp/"
UPLOAD_FOLDER = "/tmp/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global model
# Load TFLite model and allocate tensors.
# model = load_model('model.h5')
model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()

@app.route('/tmp/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template('index.html', message='No file part in the form.')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file.')
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(filename)
            file.save(filename)
            result = process_file(filename)
            file_url = url_for('static', filename=f"{file.filename}")
            return render_template('index.html', message=result, file_url=file_url)
    return render_template('index.html')

def crop_banana(image_path):
    # Read the image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Convert image to grayscale
    img_gray = ImageOps.grayscale(img)

    # Apply threshold
    img_binary = img_gray.point(lambda p: 255 if p < 150 else 0)

    # Convert to NumPy array to find contours
    img_binary_np = np.array(img_binary)

    # Label the binary image
    labeled_array, num_features = ndimage.label(img_binary_np)

    # Find contours using skimage
    contours = measure.find_contours(labeled_array, 0.8)

    # Sort contours by area (size)
    contours = sorted(contours, key=lambda x: len(x), reverse=True)

    # Get bounding box of the largest contour
    x_min = int(np.min(contours[0][:, 1]))
    x_max = int(np.max(contours[0][:, 1]))
    y_min = int(np.min(contours[0][:, 0]))
    y_max = int(np.max(contours[0][:, 0]))

    # Crop the image using bounding box coordinates
    cropped = img_np[y_min:y_max, x_min:x_max]

    return cropped

def redim(image, alto=272, largo=416):
    # Convert the NumPy array to a PIL image first
    image = Image.fromarray(image)

    # Get the dimensions of the cropped image
    old_width, old_height = image.size

    # Calculate the aspect ratio of the old and new image
    aspect_ratio = old_width / old_height

    # Pad the image to maintain aspect ratio
    if old_width < old_height * (largo / alto):
        new_width = int(old_height * (largo / alto))
        padding = (new_width - old_width) // 2
        padded = ImageOps.expand(image, (padding, 0, padding, 0), fill="black")
    else:
        new_height = int(old_width / (largo / alto))
        padding = (new_height - old_height) // 2
        padded = ImageOps.expand(image, (0, padding, 0, padding), fill="black")

    # Resize the image to the desired size
    resized = padded.resize((largo, alto), Image.LANCZOS)

    # Convert the PIL image back to a NumPy array if needed
    resized_np = np.array(resized)

    return resized_np

def process_file_(filepath):
    # Aplicar las funciones de preprocesamiento
    image = crop_banana(filepath)
    image = redim(image, alto=280, largo=450)

    # Convertir la imagen a un formato compatible con tu modelo
    image = np.expand_dims(image, axis=0)

    # Aplicar el modelo de IA para inferir el tipo de banano
    prediction = model.predict(image) # asumiendo que tienes un modelo llamado "model"

    # Decodificar la predicción
    pred = prediction[0][0]
    if pred > 0.5:
        return f"Banano para exportación"
        # return f"Banano para exportación | confidence: {round(pred, 2)}"
    else:
        return f"Banano de rechazo"
        # return f"Banano de rechazo | confidence: {round(pred, 2)}"

def process_file(filepath):
    image = crop_banana(filepath)
    image = redim(image, alto=280, largo=450).reshape(-1, 280, 450, 3)
    
    # Get input and output tensors information from the model file
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    model.set_tensor(input_details[0]['index'], image.astype(np.float32))
    model.invoke()
    
    output_data = model.get_tensor(output_details[0]['index'])
    pred = output_data[0][0]
    if pred > 0.5:
        return f"Banano para exportación"
        # return f"Banano para exportación | confidence: {round(pred, 2)}"
    else:
        return f"Banano de rechazo"
        # return f"Banano de rechazo | confidence: {round(pred, 2)}"

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5001, debug=True)
    app.debug = True
    app.run()
