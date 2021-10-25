from __future__ import absolute_import, division, print_function, unicode_literals

# imports
import os
import shutil
import cv2
import numpy as np
from PIL import Image as Image


temporary_frame_directory = "./data/temporarySpace/"
temporary_image_location = temporary_frame_directory + "image.jpg"

if not os.path.exists(temporary_frame_directory):
    os.makedirs(temporary_frame_directory)

# path to directories
data_directory = "./data/dataset/unprocessed_data/"

# <---------------------------------- Create the directory structure ---------------------------------->
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

    # CREATE ASL DIRECTORIES
    os.makedirs(data_directory + "A")
    os.makedirs(data_directory + "B")
    os.makedirs(data_directory + "C")
    os.makedirs(data_directory + "D")
    os.makedirs(data_directory + "E")
    os.makedirs(data_directory + "F")
    os.makedirs(data_directory + "G")
    os.makedirs(data_directory + "H")
    os.makedirs(data_directory + "I")
    os.makedirs(data_directory + "K")
    os.makedirs(data_directory + "L")
    os.makedirs(data_directory + "M")
    os.makedirs(data_directory + "N")
    os.makedirs(data_directory + "O")
    os.makedirs(data_directory + "P")
    os.makedirs(data_directory + "Q")
    os.makedirs(data_directory + "R")
    os.makedirs(data_directory + "S")
    os.makedirs(data_directory + "T")
    os.makedirs(data_directory + "U")
    os.makedirs(data_directory + "V")
    os.makedirs(data_directory + "W")
    os.makedirs(data_directory + "X")
    os.makedirs(data_directory + "Y")
    os.makedirs(data_directory + "EMPTY")
    os.makedirs(data_directory + "BACKSPACE")
    os.makedirs(data_directory + "SPACE")
# <---------------------------------- Create the directory structure ---------------------------------->


def current_frame():
    global gray_frame, rgb_converted_img
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    difference = cv2.absdiff(first_frame_gray, gray_frame)
    _, difference = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)

    cv2.imwrite(temporary_image_location, difference)
    im = Image.open(temporary_image_location).convert('RGB')
    npim = 255 - np.array(im)
    Image.fromarray(npim).save(temporary_image_location)

    rgb_converted_img = cv2.imread(temporary_image_location)
    cv2.imshow("RGB-converted_1", rgb_converted_img)


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

# Load the web-camera
camera = cv2.VideoCapture(0)
first_frame()

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)  # mirror the frame

    # Count of existing images in each directory
    count = {'A': len(os.listdir(data_directory + "A")),
             'B': len(os.listdir(data_directory + "B")),
             'C': len(os.listdir(data_directory + "C")),
             'D': len(os.listdir(data_directory + "D")),
             'E': len(os.listdir(data_directory + "E")),
             'F': len(os.listdir(data_directory + "F")),
             'G': len(os.listdir(data_directory + "G")),
             'H': len(os.listdir(data_directory + "H")),
             'I': len(os.listdir(data_directory + "I")),
             'K': len(os.listdir(data_directory + "K")),
             'L': len(os.listdir(data_directory + "L")),
             'M': len(os.listdir(data_directory + "M")),
             'N': len(os.listdir(data_directory + "N")),
             'O': len(os.listdir(data_directory + "O")),
             'P': len(os.listdir(data_directory + "P")),
             'Q': len(os.listdir(data_directory + "Q")),
             'R': len(os.listdir(data_directory + "R")),
             'S': len(os.listdir(data_directory + "S")),
             'T': len(os.listdir(data_directory + "T")),
             'U': len(os.listdir(data_directory + "U")),
             'V': len(os.listdir(data_directory + "V")),
             'W': len(os.listdir(data_directory + "W")),
             'X': len(os.listdir(data_directory + "X")),
             'Y': len(os.listdir(data_directory + "Y")),
             'EMPTY': len(os.listdir(data_directory + "EMPTY")),
             'BACKSPACE': len(os.listdir(data_directory + "BACKSPACE")),
             'SPACE': len(os.listdir(data_directory + "SPACE"))
             }

    # Displaying the count of each directory to the screen
    cv2.putText(frame, "NUMBER OF IMAGES", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "A : " + str(count['A']), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "B : " + str(count['B']), (10, 55), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "C : " + str(count['C']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "D : " + str(count['D']), (10, 85), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "E : " + str(count['E']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "F : " + str(count['F']), (10, 115), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "G : " + str(count['G']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "H : " + str(count['H']), (10, 145), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "I : " + str(count['I']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "K : " + str(count['K']), (10, 175), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "L : " + str(count['L']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "M : " + str(count['M']), (10, 205), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "N : " + str(count['N']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "O : " + str(count['O']), (10, 235), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "P : " + str(count['P']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "Q : " + str(count['Q']), (10, 265), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "R : " + str(count['R']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "S : " + str(count['S']), (10, 295), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "T : " + str(count['T']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "U : " + str(count['U']), (10, 325), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "V : " + str(count['V']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "W : " + str(count['W']), (10, 355), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "X : " + str(count['X']), (10, 370), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "Y : " + str(count['Y']), (10, 385), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "EMPTY : " + str(count['EMPTY']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "BACKSPACE : " + str(count['BACKSPACE']), (10, 415), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "SPACE : " + str(count['SPACE']), (10, 430), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

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

    current_frame()

    cv2.putText(frame, "Details on how to", (320, 345), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    cv2.putText(frame, "operate this script", (320, 365), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    cv2.putText(frame, "can be found in the", (320, 385), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    cv2.putText(frame, "README", (320, 405), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    cv2.putText(frame, "press ESC to exit window", (320, 440), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 255), 1)
    cv2.imshow("Frame", frame)  # display frame

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        shutil.rmtree(temporary_frame_directory)  # deletes directory and all its contents
        break
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(data_directory + 'A/A_' + str(count['A']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(data_directory + 'B/B_' + str(count['B']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(data_directory + 'C/C_' + str(count['C']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(data_directory + 'D/D_' + str(count['D']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(data_directory + 'E/E_' + str(count['E']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(data_directory + 'F/F_' + str(count['F']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(data_directory + 'G/G_' + str(count['G']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(data_directory + 'H/H_' + str(count['H']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(data_directory + 'I/I_' + str(count['I']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(data_directory + 'K/K_' + str(count['K']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(data_directory + 'L/L_' + str(count['L']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(data_directory + 'M/M_' + str(count['M']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(data_directory + 'N/N_' + str(count['N']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(data_directory + 'O/O_' + str(count['O']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(data_directory + 'P/P_' + str(count['P']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(data_directory + 'Q/Q_' + str(count['Q']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(data_directory + 'R/R_' + str(count['R']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(data_directory + 'S/S_' + str(count['S']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(data_directory + 'T/T_' + str(count['T']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(data_directory + 'U/U_' + str(count['U']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(data_directory + 'V/V_' + str(count['V']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(data_directory + 'W/W_' + str(count['W']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(data_directory + 'X/X_' + str(count['X']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(data_directory + 'Y/Y_' + str(count['Y']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == 13:
        cv2.imwrite(data_directory + 'EMPTY/EMPTY_' + str(count['EMPTY']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == 8:
        cv2.imwrite(data_directory + 'BACKSPACE/BACKSPACE_' + str(count['BACKSPACE']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == 32:
        cv2.imwrite(data_directory + 'SPACE/SPACE_' + str(count['SPACE']) + '.jpg', rgb_converted_img)
    if interrupt & 0xFF == 9:
        first_frame_gray = gray_frame

camera.release()
cv2.destroyAllWindows()
