# PROJECT: HAND-GESTURE-CONTROLLED-VEHICLE-
## *Description: It will be controlled from the laptop's camera, then it will transfer the predicted results of hand gesture actions to the server and send the results to Raspberry Pi 4.*
There are 3 tasks for you to launch my project
1. Task 1: Training model LSTM
- Run Training/make_dir.py: Making directory (path) to save .npy files (.npy files will appear when you execute next step). It will create 4 folders: forward, left, right, backward in folder Project_Data, each folder has 120 .npy files.
- Run Training/get_data.py: Collecting your key points (120 videos, 5 frames per video) and storing in .npy files
- Run Training/main.py: Creating modeling.h5, you can use it to predict RIGHT HAND gesture.
2. Task 2: Running laptop/hand_tracking.py in your laptop or computer.
3. Task 3: Running RaspberryPi/client.py in your Raspberrypi4 device.
## NOTE: Firstly you must train model to get modeling.h5, then implement task 2 and task 3 in simultaneuously. You can extract all and ultilize my Project_Data.rar file, and skip run make_dir.py, get_data.py.
