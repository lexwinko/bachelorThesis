import pandas as pd
import sys


def visualizeData(hashtag):
	predGNB = pd.read_csv("predictionGNB_"+hashtag+".csv")
	predLR = pd.read_csv("predictionLR_"+hashtag+".csv")
	predNN = pd.read_csv("predictionNN_"+hashtag+".csv")
	predRF = pd.read_csv("predictionRF_"+hashtag+".csv")
	predSVM = pd.read_csv("predictionSVM_"+hashtag+".csv")

	lang = ['tlEN', 'ruEN', 'ptEN', 'plEN', 'koEN', 'jpEN', 'itEN', 'inEN', 'hiEN', 'frEN', 'esEN', 'deEN']

	print(predGNB)
	print(predLR)

	resultGNB = [0,0,0,0,0,0,0,0,0,0,0,0]
	resultLR = [0,0,0,0,0,0,0,0,0,0,0,0]
	resultNN = [0,0,0,0,0,0,0,0,0,0,0,0]
	resultRF = [0,0,0,0,0,0,0,0,0,0,0,0]
	resultSVM = [0,0,0,0,0,0,0,0,0,0,0,0]
	row = 1
	for x in range(0,len(lang)):
		resultGNB[x] = predGNB['lang_'+str(row)].sum()
		resultNN[x] = predNN['lang_'+str(row)].sum()
		resultLR[x] = predLR['lang_'+str(row)].sum()
		resultRF[x] = predRF['lang_'+str(row)].sum()
		resultSVM[x] = predSVM['lang_'+str(row)].sum()
		row += 1

	print(resultGNB, resultNN, resultLR, resultRF, resultSVM)



if __name__ == "__main__":
	print("Checking "+sys.argv[1])
	visualizeData(sys.argv[1])