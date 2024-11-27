# ------------------------------------------------------------------------
# Trash Classifier ML Project With Teachable Machine
# Made by Aaradhita Singh on November 24th, 2024 
# --------------------------------------------------------------------------

#importing everything needed
from gpiozero import Button, AngularServo
import subprocess
from time import sleep
from tflite_runtime import interpreter
from PIL import Image
import numpy as np
import os

button = Button(14) #replace the gpio pin if needed
servo = AngularServo(15, min_pulse_width=0.0006, max_pulse_width=0.0023) #you can replace this one too
interpreter = interpreter.Interpreter(model_path="/home/aaradhita-singh/Downloads/model.tflite") #replace this filepath with the filepath to your model
interpreter.allocate_tensors()

#image prediction and moving the servo
def predict_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_array = np.expand_dims(img_array, axis=0)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        print(f"Prediction: {prediction}")

        if prediction == 0:
            servo.angle = 90
            sleep(2)
            servo.angle = 0
        else:
            servo.angle = -90
            sleep(2)
            servo.angle = 0

    except Exception as e:
        print(f"Error during prediction: {e}")

#taking the photo
def take_photo():
    try:
        output_path = '/home/aaradhita-singh/Pictures/image.jpg'  #replace this with where you want your image to go
        subprocess.run([
            'libcamera-still',
            '-o', output_path,
            '--timeout', '0.1'
        ])
        sleep(1)
        predict_image(output_path)
    except Exception as e:
        print(f"Error taking photo: {e}")

#main function
while True:
    if button.is_pressed:
        take_photo()
    sleep(0.1)
