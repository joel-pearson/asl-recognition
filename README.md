# asl-recognition
Classifying gestures of the American-Sign-Language (ASL) alphabet and writing the performed sequence to a text output in real-time. 

---

This project was undertaken as part of my Bachelors dissertation in Computer Science. 

This project classifies gestures of the ASL alphabet using a convolutional neural network in real time and displays the performed sequence. 
The system utilises background subtraction methods to isolate the hand in the frame. I also collected my own dataset using the script 'data-collection.py', which was used to train the model and can be found in the 'data/dataset' directory.

The system has been trained to classify 27 different hand gestures, which include: 
- the letter A-Z (excluding J and Z because they are not static gestures)
- empty (for when the frame contains no hand gesture)
- space (to enter a space between words in a sequence)
- backspace (to delete characters from the sequence)

---
### Installation
To run the model Python 3.7 (64-Bit) is required, the model will not work with 32-Bit installations. 

To use the ASL recognition model:
- create a new virtual environment (to avoid installing packages to base python installation).
- ensure that the virtual environment is active.
- run "pip install -r requirements.txt"	- to install all python dependencies for the project.
