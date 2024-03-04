# This is how to read a video frame by frame and to show it.
import os

print(os.getcwd())
import cv2

os.chdir('C:/Users/Lenovo/Desktop/ManAI Project/Data/Push-ups')

video = cv2.VideoCapture('Wide Grip Push up 1.mp4')
# We start with extracting the first frame from it and the status that is True if the video still has othr frames to read
status, frame = video.read()
# Then we will have a loop that will read the rest of the frames and then we will show the video.
while status:
	cv2.imshow('', frame)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break
	status, frame = video.read()
video.release()
