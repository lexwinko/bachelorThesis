import nasty
from polyglot.detect import Detector
from polyglot.text import Text
import os
import csv
import sys


tweet_stream_turkey = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_turkey = {'ArtCul' : ["#SezenAksu", "#Cemre", "#BugünGünlerdenGALATASARAY", "#BugünGünlerdenTrabzonspor"],
						'BuiTecSci' : ["#Teknofest2019", "#coronaviruesue"],
							'SocSoc' : ["#Perşembe","#salı"],
							'Pol' : ["#BaharKalkanı","#DünyanınEnGüçlüOrdusuyuz"] }

tweet_stream_france = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_france = 		{	'ArtCul' : ["#MariesAuPremierRegard", "#JeudiPhoto"],
							'BuiTecSci' : ["#CoronavirusFrance", "#ChangeNOW2020"],
							'SocSoc': ["#negrophile4life","#JeSuisVictime", "#CesarDeLaHonte"],
							'Pol': ["#49al3","#greve20fevrier"] }

tweet_stream_greece = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_greece = 		{	'ArtCul': ["#AdinamosKrikosGr", "#tokafetisxaras", "paokoly"],
							'BuiTecSci' :["#mitefgreece", "#reloadgreece"],
							'SocSoc': ["#Τσικνοπεμπτη","#28ηΟκτωβριου"],
							'Pol' :["#εβρος","#μεταναστες"] }

tweet_stream_germany = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_germany = 		{	'ArtCul': ["#AUTGER", "#DerSchwarzeSchwan"],
							'BuiTecSci': ["#spiegelonline", "#BahnCard"],
							'SocSoc': ["#Umweltsau","#Weltknuddeltag"],
							'Pol' :["#Sterbehilfe","#Bauernproteste", "dieUhrtickt"] }

tweet_stream_russia = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_russia = 		{	'ArtCul': ["#BTSTOUR2020_Russia", "#Биатлон"],
							'BuiTecSci' :[],
							'SocSoc' :[],
							'Pol': [] }

tweet_stream_japan = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_japan = 		{	'ArtCul': ["#popjwave", "#annkw"],
							'BuiTecSci': [],
							'SocSoc': [],
							'Pol' :[] }

tweet_stream_india = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_india = 		{	'ArtCul' :["#PonniyinSelvan", "#NewEra_By_SaintRampalJi"],
							'BuiTecSci' :["#IISF2019"],
							'SocSoc': ["#AskSaiTej","#Dabangg3Reviews"],
							'Pol': ["#99535_88585_AgainstCAA","#AzadiForAzad"] }

tweet_stream_native = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_native = 		{	'ArtCul' :["#titansvschiefs", "#winniethepoohday"],
							'BuiTecSci': ["#SAMESBC", "#ngcx"],
							'SocSoc': ["#NationalDressUpYourPetDay","#ThingsThatUniteUs"],
							'Pol' :["#TellTheTruthJoe","#VirginiaRally"] }

tweet_stream_worldwide = {'ArtCul':{}, 'BuiTecSci':{}, 'SocSoc':{}, 'Pol':{} }
tweet_search_worldwide =	{	'ArtCul' :["#GameOfThrones", "#BoyWithLuv"],
							'BuiTecSci': ["#CES", "#COVID2019"],
							'SocSoc': ["#loveyourpetday","#2020NewYear"],
							'Pol' :["#hanau","#InternationalWomensDay"] }

						
categories = ["ArtCul", "BuiTecSci", "SocSoc", "Pol"] 

def filter_text(txt):
	ret_list = {}
	for hashtag in txt:
		len_raw = 0
		len_filtered = 0
		ret_list[hashtag] = {'len_raw': 0, 'len_filtered': 0, 'data':[]}
		for tweet in txt[hashtag]:
			len_raw += 1
			if(Text(tweet.text).language.name == 'English'):
				len_filtered += 1
				ret_list[hashtag]['data'].append(tweet.text)
		ret_list[hashtag]['len_raw'] = len_raw
		ret_list[hashtag]['len_filtered'] = len_filtered
	return ret_list




def save_to_file(txt):
	with open('info.csv', "w") as f:
		w = csv.DictWriter(f, ['category','hashtag', 'len_raw', 'len_filtered'])
		w.writeheader()
		f.close()
	for cat in txt:
		for hashtag in txt[cat]:
			os.makedirs(os.path.dirname(cat + '/' + 'info.csv'), exist_ok=True)
			row = {'category': cat, 'hashtag': hashtag, 'len_raw':txt[cat][hashtag]['len_raw'], 'len_filtered':txt[cat][hashtag]['len_filtered']}
			with open('info.csv', "a") as f:
				w = csv.DictWriter(f, fieldnames=['category','hashtag', 'len_raw', 'len_filtered'])
				w.writerow(row)
				f.close()
			filename = cat + '/' + hashtag + '.csv'
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			with open(filename, "w", encoding='utf-8') as f:
				w = csv.DictWriter(f, ['text'])
				w.writeheader()
				for line in txt[cat][hashtag]['data']:
					w.writerow({'text': line})


if __name__ == "__main__":
	hashtag_file = ""
	data_file = ""
	if(sys.argv[1] == 'turkey'):
		hashtag_file = tweet_search_turkey
		data_file = tweet_stream_turkey
	elif(sys.argv[1] == 'france'):
		hashtag_file = tweet_search_france
		data_file = tweet_stream_france
	elif(sys.argv[1] == 'greece'):
		hashtag_file = tweet_search_greece
		data_file = tweet_stream_greece
	elif(sys.argv[1] == 'germany'):
		hashtag_file = tweet_search_germany
		data_file = tweet_stream_germany
	elif(sys.argv[1] == 'russia'):
		hashtag_file = tweet_search_russia
		data_file = tweet_stream_russia
	elif(sys.argv[1] == 'japan'):
		hashtag_file = tweet_search_japan
		data_file = tweet_stream_japan
	elif(sys.argv[1] == 'india'):
		hashtag_file = tweet_search_india
		data_file = tweet_stream_india
	elif(sys.argv[1] == 'native'):
		hashtag_file = tweet_search_native
		data_file = tweet_stream_native
	elif(sys.argv[1] == 'worldwide'):
		hashtag_file = tweet_search_worldwide
		data_file = tweet_stream_worldwide

	for cat in categories:
		for hashtag in hashtag_file[cat]:
			data_file[cat][hashtag] = nasty.Search(hashtag, lang="en").request()
		data_file[cat] = filter_text(data_file[cat])
	save_to_file(data_file)





	


	

