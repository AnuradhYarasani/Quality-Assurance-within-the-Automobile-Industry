# Import required libraries
import flask
from flask import Flask, render_template,request 
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import warnings
from tensorflow.keras.preprocessing import image
from PIL import Image
# filter warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(filename='log/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/defect_detection', methods = ['POST'])
def defect_detection():    
    if flask.request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
            
        if file.filename == '':
            return "No selected file"

   
        try:
            BATCH_SIZE = 8 
            IMG_HEIGHT = 90
            IMG_WIDTH = 90
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal",
                                  input_shape=(IMG_HEIGHT,
                                              IMG_WIDTH,
                                              3))    ]    )
            num_classes = 2

            model = Sequential([
              data_augmentation,
              layers.Conv2D(16, 3, padding='same', activation='relu'),
              layers.MaxPooling2D(),
              layers.Conv2D(32, 3, padding='same', activation='relu'),
              layers.MaxPooling2D(),
              layers.Conv2D(64, 3, padding='same', activation='relu'),
              layers.MaxPooling2D(),
              layers.Dropout(0.2),
              layers.Flatten(),
              layers.Dense(128, activation='relu'),
              layers.Dense(num_classes, name="outputs")
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.summary()
            model.load_weights('Models/aircraft_model_weights.h5')
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img = image.load_img(full_path, target_size=(IMG_HEIGHT,IMG_WIDTH))  

            # Convert the image to a numpy array
            img_array = image.img_to_array(img)

            # Expand the dimensions to match the shape expected by the model (usually batch size of 1)
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)

            # Optionally, decode the predictions if your model uses one-hot encoding for classification
            # For example, if it's a classification model:
            class_labels = ['Defects', 'Normal']  # Replace with your actual class labels
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]
            print("Predicted class:", predicted_class)
            return "Predicted class: "+ predicted_class
        except Exception as e:
            logging.exception("Exception occurred")
            return " "


        
        
if __name__ == '__main__':
    app.run("127.0.0.1",5009,debug=True)