from __future__ import absolute_import, division, print_function, unicode_literals

# imports
import os
import cv2
import shutil

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
import numpy as np
from PIL import Image as Image


temporary_frame_directory = "./data/framePrediction/"
temporary_image_location = temporary_frame_directory + "image.jpg"

# Loads the .h5 file and stores to a variable
loaded_model = keras.models.load_model('./model/model-bs.h5', custom_objects={'KerasLayer': hub.KerasLayer})
loaded_model.summary()  # Displays a summary of the loaded_model

IMAGE_SHAPE = (224, 224)  # assigns a size to the variable IMAGE_SHAPE

if not os.path.exists(temporary_frame_directory):
    os.makedirs(temporary_frame_directory)


def prediction():
    global predicted_class_name
    image = temporary_image_location  # path to file for testing model
    image = Image.open(image).resize(IMAGE_SHAPE)  # opens image and resizes to the correct dimensions for the model

    # Places the image in an array
    image = np.array(image) / 255.0

    # Adds a batch dimension and passes the image to the model
    result = loaded_model.predict(image[np.newaxis, ...])

    # Finds the top rated probability for the prediction
    predicted_class = np.argmax(result[0], axis=-1)

    # Gets the label file saved as a .txt
    labels_path = './data/labels.txt'
    character_labels = np.array(open(labels_path).read().splitlines())

    predicted_class_name = character_labels[predicted_class]

    # Displaying the predictions
    pred = cv2.putText(frame, "Prediction: ", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)
    cv2.putText(frame, predicted_class_name, (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 1)
    cv2.putText(frame, "press 'spacebar' to reset background", (10, 420), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
    cv2.putText(frame, "press ESC to exit window", (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
    cv2.imshow("Prediction", pred)


def current_frame():
    global gray_frame
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    difference = cv2.absdiff(first_frame_gray, gray_frame)
    _, difference = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)

    cv2.imwrite(temporary_image_location, difference)
    im = Image.open(temporary_image_location).convert('RGB')
    npim = 255 - np.array(im)
    Image.fromarray(npim).save(temporary_image_location)

    rgb_converted_img = cv2.imread(temporary_image_location)
    cv2.imshow("RGB-converted", rgb_converted_img)


def first_frame():
    global first_frame_gray
    _, first_frame = camera.read()
    first_frame = cv2.flip(first_frame, 1)  # mirror the frame

    # Coordinates of the ROI
    x1 = int(0.5 * first_frame.shape[1])
    y1 = 10
    x2 = first_frame.shape[1] - 10
    y2 = int(0.5 * first_frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(first_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    first_roi = first_frame[y1:y2, x1:x2]
    first_roi = cv2.resize(first_roi, (224, 224))

    first_frame_gray = cv2.cvtColor(first_roi, cv2.COLOR_BGR2GRAY)
    first_frame_gray = cv2.GaussianBlur(first_frame_gray, (5, 5), 0)


camera = cv2.VideoCapture(0)
frame_count = 0  # set frame_count to 0
current_predicted_class = ""  # initiate variable 'current_predicted_class'
string = []  # create array called 'string'
first_frame()

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)  # mirror the frame

    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (224, 224))

    # write array to text file named output.txt
    with open("data/output.txt", "w") as txt_file:
        for i in string:
            txt_file.write("".join(i))

    # read output.txt
    output_text = open("data/output.txt", "r")
    text_file = output_text.read()

    background_img = cv2.imread("./data/black-background.jpg")
    background_img = cv2.resize(background_img, (500, 200))
    cv2.putText(background_img, text_file, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
    cv2.imshow("Sequence", background_img)

    cv2.imwrite(temporary_image_location, roi)  # write frame of ROI to image location

    current_frame()
    prediction()

    # if statements for storing predicted classes to the array
    if predicted_class_name == "A":
        if current_predicted_class != "A":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):  # frame_count can get to 40 or 80 to account for double letters
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "B":
        if current_predicted_class != "B":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "C":
        if current_predicted_class != "C":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "D":
        if current_predicted_class != "D":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "E":
        if current_predicted_class != "E":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "F":
        if current_predicted_class != "F":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "G":
        if current_predicted_class != "G":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "H":
        if current_predicted_class != "H":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "I":
        if current_predicted_class != "I":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "K":
        if current_predicted_class != "K":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "L":
        if current_predicted_class != "L":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "M":
        if current_predicted_class != "M":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "N":
        if current_predicted_class != "N":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "O":
        if current_predicted_class != "O":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "P":
        if current_predicted_class != "P":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "Q":
        if current_predicted_class != "Q":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "R":
        if current_predicted_class != "R":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "S":
        if current_predicted_class != "S":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "T":
        if current_predicted_class != "T":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "U":
        if current_predicted_class != "U":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "V":
        if current_predicted_class != "V":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "W":
        if current_predicted_class != "W":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "X":
        if current_predicted_class != "X":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "Y":
        if current_predicted_class != "Y":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(predicted_class_name)
            print(string)

    if predicted_class_name == "SPACE":
        if current_predicted_class != "SPACE":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.append(" ")
            print(string)

    if predicted_class_name == "BACKSPACE":
        if current_predicted_class != "BACKSPACE":
            current_predicted_class = predicted_class_name
            frame_count = 0
        else:
            pass
        frame_count = frame_count + 1
        if (frame_count == 40) or (frame_count == 80):
            string.pop()
            print(string)

    key = cv2.waitKey(10)
    if key & 0xFF == 27:  # esc key
        shutil.rmtree(temporary_frame_directory)  # deletes directory and all its contents
        break
    if key & 0xFF == 32:  # space-bar key
        first_frame_gray = gray_frame

camera.release()
cv2.destroyAllWindows()
