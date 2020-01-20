import pandas as pd
import sys
import csv
import os
import glob
import numpy as np
from langdetect import detect

def sumCSV(file):
	nlpFile = pd.read_csv(file, header=0, sep=',')

	used_features = [ 'len', '#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V','elong']
	#export_features = [ '#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V','elong']

	#print(nlpFile)
	meanFile = nlpFile[used_features].mean().round(decimals=3)
	
	print(meanFile)

	with open('summedResult.csv', "w") as f:
		w = csv.writer(f)
		w.writerow(used_features)
		w.writerow(meanFile)

def fixCSV(file):
	csvFile = pd.read_csv(file, header=0, sep=';')
	fixedCSV = []

	#print(csvFile)

	print(len(csvFile))

	for row in csvFile.values:
		text = row[14]
		if not 'RT' in text and detect(text) == 'en' and not any(text in sl for sl in fixedCSV):
			fixedCSV.append(row)

	
	print(len(fixedCSV))

	outputFile = pd.DataFrame(fixedCSV, columns=csvFile.columns)
	outputFile.to_csv("fixedCSV.csv", sep=';', index=False, encoding='utf-8-sig')


def concatCSV(folder, type, name):
	combined_csv = pd.DataFrame()
	if(type == 0):
		os.chdir(folder)
		extension = 'csv'
		all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
		print(all_filenames)
		#combine all files in the list
		print(folder)
		for f in all_filenames:
			for chunk in pd.read_csv(f, header=0, chunksize=10000, skiprows= lambda x: x> 0 and np.random.rand() > 0.50, nrows=20000):
				#print(chunk)
				combined_csv = pd.concat([combined_csv, chunk])

	elif(type == 1):
		for file in folder:
			for chunk in pd.read_csv(file, header=0, chunksize=10000, skiprows= lambda x: x> 0 and np.random.rand() > 0.50, nrows=20000):
				#print(chunk)
				combined_csv = pd.concat([combined_csv, chunk])
	print(combined_csv)
	#pd.concat([pd.concat(pd.read_csv(f, header=0, chunksize= 100,sep=',')) for f in all_filenames ])
	#export to csv
	combined_csv.to_csv( name + ".csv", sep=',', index=False, encoding='utf-8-sig')



if __name__ == "__main__":
	deftype = sys.argv[1]
	file = sys.argv[2]
	if(len(sys.argv) > 3):
		file2 = sys.argv[3]
	print('Input '+ file)
	if(deftype == 'sum'):
		sumCSV(file)
	elif(deftype == 'concfold'):
		concatCSV(file, type=0, name="combinedFolder")
	elif(deftype == 'concfile'):
		if(len(sys.argv) > 4):
			name = sys.argv[4]
		else:
			name = "combinedFiles"
		concatCSV([file,file2], type=1, name=name)
	elif(deftype == 'fix'):
		fixCSV(file)
	print('Done')
	#print(result)