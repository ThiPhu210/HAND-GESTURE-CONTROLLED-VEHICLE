import os
DATA_PATH = 'C:\\NUMBER_GESTURE\\Project_Data'
ACTIONS = ['forward', 'left', 'right','backward']
video_num = 120

def make_data_path() :
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            os.makedirs(action_path)
        
        for i in range(0, video_num):
            video_path = os.path.join(action_path, str(i))
            if not os.path.exists(video_path):
                os.makedirs(video_path)



make_data_path()