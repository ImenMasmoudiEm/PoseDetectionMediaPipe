# This is how to read a video frame by frame and to extract the key points from the posture and store it in a csv file.
import os
import cv2
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#We will be needing the names of the Keypoints coordinates later to create the dasets, that is why we created it using this methode.
L = ''
for i in range(33):
	L += ' x' + str(i) + ' y' + str(i) + ' z' + str(i)
L = L.split()

print(os.getcwd())
#os.chdir('C:/Users/Lenovo/Desktop/ManAI Project/Data/Push-ups')
folderPath = 'C:/Users/Lenovo/Desktop/ManAI Project/Transformed Data/'

motions = ['Push ups', 'Barbell Curl', 'Squat']
#In the big for loop, we will go through the three motions' videos
for i1 in range(3):
	folderPath = os.path.join('C:/Users/Lenovo/Desktop/ManAI Project/Transformed Data/', motions[i1])
	os.chdir(folderPath)
	#In the second for loop we will go through all of the videos in the motion directory one by one
	for i2 in tqdm(os.listdir(folderPath)):
		video = cv2.VideoCapture(i2)
		status, frame = video.read()
		total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
		keypoints = {i: [] for i in L}
		D = []
		print(i2)
		with mp_pose.Pose(
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5) as pose:
			while status:
				cv2.imshow('', frame)
				results = pose.process(frame)

				for i in range(33):
					D.append(results.pose_landmarks.landmark[i])
				status, frame = video.read()
			#After extracting the pose_landmarks in all of the frames in the video, we will be extracting the coordinates and adding them to the dictionary keypoints.
			for i in range(int(total_frames)):
				for j in range(99):
					if j % 3 == 0:
						keypoints[L[j]].append(D[(i * 33) + (j % 33)].x)
					elif j % 3 == 1:
						keypoints[L[j]].append(D[(i * 33) + (j % 33)].y)
					else:
						keypoints[L[j]].append(D[(i * 33) + (j % 33)].z)
		video.release()
		#To save the data as a csv file, we transformed it to a DataFrame object first and then saved it as a csv file.
		Keypoints = pd.DataFrame(keypoints)
		#Keypoints.to_csv(i2 + '.csv', index=False)
		print(f"Total Frames: {total_frames}")