import os
import cv2
import math as m
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
import tensorflow as tf
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

plt.rcParams['figure.figsize'] = [10, 10]
df = pd.read_csv('C:/Users/Lenovo/Desktop/TC/ManAI Project/Ressembled Data/Totaldf.csv')


def scaler(x1, y1, z1, x2, y2, z2):
	"""This function calculates the scalar product between two vectors"""
	return x1 * x2 + y1 * y2 + z1 * z2


def Length(x1, y1, z1, x2, y2, z2):
	"""This function calculates the distance between two points"""
	return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def Angle(x1, y1, z1, x2, y2, z2, x3, y3, z3):
	"""This function calculates the Angle between two vectors that are represented by three points:
	Point 1 having (x1,y1,z1) as coordinates
	Point 2 having (x2,y2,z2) as coordinates
	Point 3 having (x3,y3,z3) as coordinates
	The angle is between the vectors:
	Vector 1: from point 1 to the point 2
	Vector 2: from point 1 to the point 3"""
	return m.acos((scaler(x2 - x1, y2 - y1, z2 - z1, x3 - x2, y3 - y2, z3 - z2)) / (
			Length(x1, y1, z1, x2, y2, z2) * Length(x2, y2, z2, x3, y3, z3)))


def listCreation(x1, y1, z1, x2, y2, z2, x3, y3, z3, fn):
	""""While building the Angles, the function Angle
  doesn't do the broadcasting of the dimensions automatically,
  that is why this function was necessary"""
	L = []
	for i in range(len(x1)):
		L.append(fn(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i], x3[i], y3[i], z3[i]))
	return L


df = pd.read_csv('C:/Users/Lenovo/Desktop/TC/ManAI Project/Ressembled Data/Totaldf.csv')

df['Target'] = df['Target'].replace(0, 2)
df['Target'] = df['Target'].replace(1, 0)
df['Target'] = df['Target'].replace(2, 1)

columnsToKeep = ['x0', 'y0', 'z0', 'x11', 'y11', 'z11', 'x12', 'y12', 'z12', 'x13', 'y13', 'z13', 'x14', 'y14', 'z14',
                 'x15', 'y15', 'z15', 'x16', 'y16', 'z16', 'y23', 'z23', 'x24', 'y24', 'z24', 'x25', 'y25', 'z25',
                 'x26', 'y26', 'z26', 'x27', 'y27', 'z27', 'x28', 'y28', 'z28']

Ndf = pd.DataFrame()
for i in columnsToKeep:
	Ndf[i] = df[i]

X = Ndf
y = df['Target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

DT = tree.DecisionTreeClassifier()
DT = DT.fit(X_train, y_train)
y_pred = DT.predict(X_valid)

print(accuracy_score(y_true=y_valid, y_pred=y_pred))
print(confusion_matrix(y_true=y_valid, y_pred=y_pred))
print(classification_report(y_true=y_valid, y_pred=y_pred, labels=[0, 1]))

columnsToKeep = ['x0', 'y0', 'z0', 'x11', 'y11', 'z11', 'x12', 'y12', 'z12', 'x13', 'y13', 'z13', 'x14', 'y14', 'z14',
                 'x15', 'y15', 'z15', 'x16', 'y16', 'z16', 'y23', 'z23', 'x24', 'y24', 'z24', 'x25', 'y25', 'z25',
                 'x26', 'y26', 'z26', 'x27', 'y27', 'z27', 'x28', 'y28', 'z28']


"""df = pd.read_csv('C:/Users/Lenovo/Desktop/TC/ManAI Project/Ressembled Data/Totaldf.csv')

plt.figure(1, figsize=(15, 5))
sns.countplot(y='Target', data=df)
plt.show()

Ndf = pd.DataFrame()
for i in columnsToKeep:
	Ndf[i] = df[i]

Ndf['Is_Vertical'] = df['Target']
Ndf['Is_Vertical'] = Ndf['Is_Vertical'].replace(0, 2)
Ndf['Is_Vertical'] = Ndf['Is_Vertical'].replace(1, 0)
Ndf['Is_Vertical'] = Ndf['Is_Vertical'].replace(2, 1)

# RE: Right Elbow
Ndf['RE'] = listCreation(df['x11'], df['y11'], df['z11'], df['x13'], df['y13'], df['z13'], df['x15'], df['y15'],
                         df['z15'], Angle)
# LE: Left Elbow
Ndf['LE'] = listCreation(df['x12'], df['y12'], df['z12'], df['x14'], df['y14'], df['z14'], df['x16'], df['y16'],
                         df['z16'], Angle)
# RA: Right Arm
Ndf['RA'] = listCreation(df['x13'], df['y13'], df['z13'], df['x11'], df['y11'], df['z11'], df['x23'], df['y23'],
                         df['z23'], Angle)
# LA: Left Arm
Ndf['LA'] = listCreation(df['x14'], df['y14'], df['z14'], df['x12'], df['y12'], df['z12'], df['x24'], df['y24'],
                         df['z24'], Angle)
# RK: Right Knee
Ndf['RK'] = listCreation(df['x23'], df['y23'], df['z23'], df['x25'], df['y25'], df['z25'], df['x27'], df['y27'],
                         df['z27'], Angle)
# LK: Left Knee
Ndf['LK'] = listCreation(df['x24'], df['y24'], df['z24'], df['x26'], df['y26'], df['z26'], df['x28'], df['y28'],
                         df['z28'], Angle)
# L: Legs
Ndf['L'] = listCreation(df['x25'], df['y25'], df['z25'], (df['x23'] + df['x24']) / 2, (df['y23'] + df['y24']) / 2,
                        (df['z23'] + df['z24']) / 2, df['x26'], df['y26'], df['z26'], Angle)

Ndf['Target'] = df['Target']

NNdf = pd.DataFrame()
c = 0
i = 0
while c < 2000:
	if Ndf.iloc[i]['Target'] == 2:
		c += 1
	print('first loop is working ')
	print(i)
	i += 1
NNdf = pd.concat([NNdf, pd.DataFrame(Ndf.iloc[0:i])])

plt.figure(1, figsize=(15, 5))
sns.countplot(y='Target', data=NNdf)
plt.show()

while i < len(Ndf):
	if Ndf.iloc[i]['Target'] != 2:
		NNdf = pd.concat([NNdf, pd.DataFrame(Ndf.iloc[i])])
	print('second loop is working ')
	print(i)
	i += 1

plt.figure(1, figsize=(15, 5))
sns.countplot(y='Target', data=NNdf)
plt.show()

X = NNdf.drop(['Target'], axis=1)
y = NNdf['Target']

x_train1, x_valid1, y_train1, y_valid1 = train_test_split(X, y, test_size=0.3, random_state=42)

x_train1 = np.expand_dims(x_train1, -1)
x_valid1 = np.expand_dims(x_valid1, -1)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=(46, )),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(3, activation='softmax'),
])

print(model.summary())
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train1, y_train1, epochs=100, verbose=1, validation_data=(x_valid1, y_valid1))

y_pred1 = model.predict(x_valid1)
y_pred1 = np.argmax(y_pred1, axis=1)

print(accuracy_score(y_true=y_valid1, y_pred=y_pred1))
print(confusion_matrix(y_true=y_valid1, y_pred=y_pred1))
print(classification_report(y_true=y_valid1, y_pred=y_pred1, labels=[0, 1, 2]))
"""

#FM = load('C:/Users/Lenovo/Desktop/ManAI Project/Models/FinalModel.joblib')
model = tf.keras.models.load_model('C:/Users/Lenovo/Desktop/TC/ManAI Project/Models/FM.h5')
L = ''
for i in range(33):
	L += ' x' + str(i) + ' y' + str(i) + ' z' + str(i)
L = L.split()
Lables = {0: 'Barbell Curl', 1: 'Push-up', 2: 'Squat'}
# describe the type of font
# to be used.
font = cv2.FONT_HERSHEY_SIMPLEX

folderPath = 'C:/Users/Lenovo/Desktop/TC/ManAI Project/Testing Data/Testing Poses'
os.chdir(folderPath)
motions = tqdm(os.listdir(folderPath))

for i1 in tqdm(os.listdir(folderPath)):
	cap = cv2.VideoCapture(i1)
	success, image = cap.read()
	Ys = []
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		while success:
			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			results = pose.process(image)
			df = {i: [] for i in L}
			Ndata = pd.DataFrame()
			D = []
			for i in range(33):
				D.append(results.pose_landmarks.landmark[i])
			for j in range(33):
				df[L[j * 3]].append(D[j].x)
				df[L[j * 3 + 1]].append(D[j].y)
				df[L[j * 3 + 2]].append(D[j].z)
			df = pd.DataFrame(df)
			for i in columnsToKeep:
				Ndata[i] = df[i]
			Ndata['Is_Vertical'] = DT.predict(Ndata)
			# RE: Right Elbow
			Ndata['RE'] = listCreation(df['x11'], df['y11'], df['z11'], df['x13'], df['y13'], df['z13'], df['x15'],
			                           df['y15'], df['z15'], Angle)
			# LE: Left Elbow
			Ndata['LE'] = listCreation(df['x12'], df['y12'], df['z12'], df['x14'], df['y14'], df['z14'], df['x16'],
			                           df['y16'], df['z16'], Angle)
			# RA: Right Arm
			Ndata['RA'] = listCreation(df['x13'], df['y13'], df['z13'], df['x11'], df['y11'], df['z11'], df['x23'],
			                           df['y23'], df['z23'], Angle)
			# LA: Left Arm
			Ndata['LA'] = listCreation(df['x14'], df['y14'], df['z14'], df['x12'], df['y12'], df['z12'], df['x24'],
			                           df['y24'], df['z24'], Angle)
			# RK: Right Knee
			Ndata['RK'] = listCreation(df['x23'], df['y23'], df['z23'], df['x25'], df['y25'], df['z25'], df['x27'],
			                           df['y27'], df['z27'], Angle)
			# LK: Left Knee
			Ndata['LK'] = listCreation(df['x24'], df['y24'], df['z24'], df['x26'], df['y26'], df['z26'], df['x28'],
			                           df['y28'], df['z28'], Angle)
			# L: Legs
			Ndata['L'] = listCreation(df['x25'], df['y25'], df['z25'], (df['x23'] + df['x24']) / 2,
			                          (df['y23'] + df['y24']) / 2, (df['z23'] + df['z24']) / 2, df['x26'],
			                          df['y26'],
			                          df['z26'], Angle)
			y_pred = model.predict(Ndata)
			y_pred = np.argmax(y_pred, axis=1)
			image.flags.writeable = True
			# Use putText() method for
			# inserting text on video
			cv2.putText(image,
			            Lables[y_pred[0]],
			            (50, 50),
			            font, 1,
			            (0, 255, 255),
			            2,
			            cv2.LINE_4)
			# Seeing the New image
			cv2.imshow('video', image)
			success, image = cap.read()
			if cv2.waitKey(5) & 0xFF == 27:
				break
	cap.release()
	cv2.destroyAllWindows()
#If the results of the model are good we can enter 1 to save the model
"""Input = int(input('Do you like the model? '))
if Input == 1:
	model.save('C:/Users/Lenovo/Desktop/ManAI Project/Models/FM1.h5')"""
