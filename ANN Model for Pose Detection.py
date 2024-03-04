import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import math as m


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
  Vector 1: from point 1 to the the point 2
  Vector 2: from point 1 to the the point 3"""
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


plt.rcParams['figure.figsize'] = [10, 10]
df = pd.read_csv('C:/Users/Lenovo/Desktop/ManAI Project/Ressembled Data/Totaldf.csv')

"""The data that we will be using to train our model will be composed of the features to keep and the features that 
we have engineered, the Is_Vertical and the Angles."""

columnsToKeep = ['x0', 'y0', 'z0', 'x11', 'y11', 'z11', 'x12', 'y12', 'z12', 'x13', 'y13', 'z13', 'x14', 'y14', 'z14',
                 'x15', 'y15', 'z15', 'x16', 'y16', 'z16', 'y23', 'z23', 'x24', 'y24', 'z24', 'x25', 'y25', 'z25',
                 'x26', 'y26', 'z26', 'x27', 'y27', 'z27', 'x28', 'y28', 'z28']

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

X = Ndf
y = df['Target']

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
print(x_train.shape)
x_train = np.expand_dims(x_train, -1)
x_valid = np.expand_dims(x_valid, -1)
print(x_train.shape)
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=(46,)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(3, activation='softmax')
])

print(model.summary())
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid))

y_pred = model.predict(x_valid)
y_pred = np.argmax(y_pred, axis=1)

print(accuracy_score(y_true=y_valid, y_pred=y_pred))
print(confusion_matrix(y_true=y_valid, y_pred=y_pred))
print(classification_report(y_true=y_valid, y_pred=y_pred, labels=[0, 1, 2]))
