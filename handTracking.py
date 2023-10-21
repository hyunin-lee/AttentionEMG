import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib as plt


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from imutils.video import FPS


from numpy import arccos, array
from numpy.linalg import norm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename','-fn', type=str)
args = parser.parse_args()

params = vars(args)  # convert args to dirctionary

# For webcam input:

#datalength = 4096*2
""""
change video_src and save_name
"""

#video_src = "./data/data0525/lhi/0525_exp_lhi_random_7.MOV"
#save_name = "./data/data0525/lhi/convertdata/test/0525_angles_exp_lhi_7.npy"
video_src = "./data/data0702/khj/"+params["filename"]+'.MOV'
save_name = "./data/data0702/khj/convertdata/angles_"+params["filename"]+'.npy'
saveData = False


clip = VideoFileClip(video_src)
print('clip duration from cv2 : ',clip.duration)

def theta(v, w): return arccos(v.dot(w)/(norm(v)*norm(w)))

def convertData(f_deque):
  Totaldata = np.zeros((3,21,len(f_deque)))
  for idx1,landmarks in enumerate(f_deque) :
    data = np.zeros((3, 21))
    for idx2,landmark in enumerate(landmarks) :
      data[0, idx2] = landmark.x
      data[1, idx2] = landmark.y
      data[2, idx2] = landmark.z
    Totaldata[:,:,idx1] = data

  return Totaldata

def convertAngle(f_array,time_list): #f_array is (3,2,time)
  datas = np.zeros((14+1,f_array.shape[2]))
  for time in range(f_array.shape[2]) :
    #data=np.zeros((3,1))(maxlen=3)
    #1st finger
    datas[0, time] = theta(f_array[:, 1, time] - f_array[:, 2, time], f_array[:, 3, time] - f_array[:, 2, time])
    datas[1, time] = theta(f_array[:, 2, time] - f_array[:, 3, time], f_array[:, 4, time] - f_array[:, 3, time])

    #2nd finger
    datas[2, time] = theta(f_array[:, 0, time] - f_array[:, 5, time], f_array[:, 6, time] - f_array[:, 5, time])
    datas[3, time] = theta(f_array[:, 5, time] - f_array[:, 6, time], f_array[:, 7, time] - f_array[:, 6, time])
    datas[4, time] = theta(f_array[:, 6, time] - f_array[:, 7, time], f_array[:, 8, time] - f_array[:, 7, time])

    #3rd finger
    datas[5, time] = theta(f_array[:, 0, time] - f_array[:, 9, time], f_array[:, 10, time] - f_array[:, 9, time])
    datas[6, time] = theta(f_array[:, 9, time] - f_array[:, 10, time], f_array[:, 11, time] - f_array[:, 10, time])
    datas[7, time] = theta(f_array[:, 10, time] - f_array[:, 11, time], f_array[:, 12, time] - f_array[:, 11, time])

    #4th finger
    datas[8, time] = theta(f_array[:, 0, time] - f_array[:, 13, time], f_array[:, 14, time] - f_array[:, 13, time])
    datas[9, time] = theta(f_array[:, 13, time] - f_array[:, 14, time], f_array[:, 15, time] - f_array[:, 14, time])
    datas[10, time] = theta(f_array[:, 14, time] - f_array[:, 15, time], f_array[:, 16, time] - f_array[:, 15, time])

    #5th finger
    datas[11, time] = theta(f_array[:, 0, time] - f_array[:, 17, time], f_array[:, 18, time] - f_array[:, 17, time])
    datas[12, time] = theta(f_array[:, 17, time] - f_array[:, 18, time], f_array[:, 19, time] - f_array[:, 18, time])
    datas[13, time] = theta(f_array[:, 18, time] - f_array[:, 19, time], f_array[:, 20, time] - f_array[:, 19, time])
  datas[14] = np.array(time_list)
  return datas




def saveTrajectory(file):
  import pickle
  with open(save_name,'wb') as f :
    pickle.dump(file,f)
#video_src=
cap = cv2.VideoCapture(video_src)
#fps = FPS().start()

video_fps = 240 #239.88 #240.07

handJointpoints = [] #deque(maxlen = datalength)
framecount = 0
time = []

with mp_hands.Hands(
    max_num_hands= 1,
    min_detection_confidence=0.001,
    min_tracking_confidence=0.001) as hands:
  while cap.isOpened():
    try :
      success, image = cap.read()
      (H, W) = image.shape[:2]
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue


      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = hands.process(image)

      #fps.update()
      #fps.stop()

      framecount += 1
      duration = framecount / video_fps

      info = [
        ("Success", "Yes" if success else "No"),
        #("FPS", "{:.2f}".format(fps.fps()))
        ("Time", "{:.2f}".format(duration))

      ]
      for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(image, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      time.append(duration)

      cv2.imshow('MediaPipe Hands', image)

      handJointpoints.append(hand_landmarks.landmark)
      #print(handJointpoints)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    except :
      print('camera end. duration from framcound : ', duration)
      cap.release()
      print('camera release')

data = convertData(handJointpoints)
data_angle = convertAngle(data,time)
if saveData :
  saveTrajectory(data_angle)

if saveVideo :
  makeVideo(data_angle)

print(data_angle)



