import pandas as pd
from joblib import dump
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = [10, 10]

df = pd.read_csv('C:/Users/Lenovo/Desktop/ManAI Project/Ressembled Data/Totaldf.csv')

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

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)

print(accuracy_score(y_true=y_valid, y_pred=y_pred))
print(confusion_matrix(y_true=y_valid, y_pred=y_pred))
print(classification_report(y_true=y_valid, y_pred=y_pred, labels=[0, 1]))

dump(clf, 'C:/Users/Lenovo/Desktop/ManAI Project/Models/DTModel.joblib')
