

folderPath = 'C:/Users/Lenovo/Desktop/ManAI Project/CSV Data/'

motions = ['CSV Files Barbell', 'CSV Files Push-ups', 'CSV Files Squat']

from tqdm import tqdm
import pandas as pd
import os

os.chdir(folderPath)
df = pd.read_csv('df.csv')


for i1 in range(3):
	folderPath1 = folderPath + motions[i1]
	os.chdir(folderPath1)
	dff = pd.read_csv('C:/Users/Lenovo/Desktop/ManAI Project/CSV Data/df.csv')
	for i2 in tqdm(os.listdir(folderPath1)):
		df1 = pd.read_csv(i2)
		df1['Target'] = [i1] * df1.shape[0]
		dff = pd.concat([dff, df1])
	df = pd.concat([df, dff])
	dff.to_csv('df.csv', index=False)

os.chdir(folderPath)
df.to_csv('Totaldf.csv', index=False)
