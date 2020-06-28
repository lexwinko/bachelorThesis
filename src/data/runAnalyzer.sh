#!/bin/sh
python3 textAnalyzer.py ../../data/processed/France/ArtCul/ twitter Romance French ArtCul
python3 textAnalyzer.py ../../data/processed/Germany/ArtCul/ twitter Germanic German ArtCul
python3 textAnalyzer.py ../../data/processed/Greece/ArtCul/ twitter Greek Greek ArtCul
python3 textAnalyzer.py ../../data/processed/India/ArtCul/ twitter Indo-Aryan Indian ArtCul
python3 textAnalyzer.py ../../data/processed/Japan/ArtCul/ twitter Japonic Japanese ArtCul
python3 textAnalyzer.py ../../data/processed/Russia/ArtCul/ twitter Balto-Slavic Russian ArtCul
python3 textAnalyzer.py ../../data/processed/Turkey/ArtCul/ twitter Turkic Turkish ArtCul
python3 textAnalyzer.py ../../data/processed/Native/ArtCul/ twitter Germanic English ArtCul
python3 textAnalyzer.py ../../data/processed/Worldwide/ArtCul/ twitter Worldwide Worldwide ArtCul
cat output/result_English_ArtCul.csv output/result_German_ArtCul.csv output/result_French_ArtCul.csv output/result_Greek_ArtCul.csv output/result_Indian_ArtCul.csv output/result_Japanese_ArtCul.csv output/result_Russian_ArtCul.csv output/result_Turkish_ArtCul.csv output/result_Worldwide_ArtCul.csv > output/combined_ArtCul.csv
rm output/result_*
#python3 ../tools/filterCSV.py output/combined_ArtCul.csv split 20

python3 textAnalyzer.py ../../data/processed/France/SocSoc/ twitter Romance French SocSoc
python3 textAnalyzer.py ../../data/processed/Germany/SocSoc/ twitter Germanic German SocSoc
python3 textAnalyzer.py ../../data/processed/Greece/SocSoc/ twitter Greek Greek SocSoc
python3 textAnalyzer.py ../../data/processed/India/SocSoc/ twitter Indo-Aryan Indian SocSo
python3 textAnalyzer.py ../../data/processed/Turkey/SocSoc/ twitter Turkic Turkish SocSoc
python3 textAnalyzer.py ../../data/processed/Native/SocSoc/ twitter Germanic English SocSoc
python3 textAnalyzer.py ../../data/processed/Worldwide/SocSoc/ twitter Worldwide Worldwide SocSoc
cat output/result_English_SocSoc.csv output/result_German_SocSoc.csv output/result_French_SocSoc.csv output/result_Greek_SocSoc.csv output/result_Indian_SocSoc.csv output/result_Turkish_SocSoc.csv output/result_Worldwide_SocSoc.csv > output/combined_SocSoc.csv
rm output/result_*
#python3 ../tools/filterCSV.py output/combined_SocSoc.csv split 20

python3 textAnalyzer.py ../../data/processed/France/BuiTecSci/ twitter Romance French BuiTecSci
python3 textAnalyzer.py ../../data/processed/Germany/BuiTecSci/ twitter Germanic German BuiTecSci
python3 textAnalyzer.py ../../data/processed/Greece/BuiTecSci/ twitter Greek Greek BuiTecSci
python3 textAnalyzer.py ../../data/processed/India/BuiTecSci/ twitter Indo-Aryan Indian BuiTecSci
python3 textAnalyzer.py ../../data/processed/Turkey/BuiTecSci/ twitter Turkic Turkish BuiTecSci
python3 textAnalyzer.py ../../data/processed/Native/BuiTecSci/ twitter Germanic English BuiTecSci
python3 textAnalyzer.py ../../data/processed/Worldwide/BuiTecSci/ twitter Worldwide Worldwide BuiTecSci
cat output/result_English_BuiTecSci.csv output/result_German_BuiTecSci.csv output/result_French_BuiTecSci.csv output/result_Greek_BuiTecSci.csv output/result_Indian_BuiTecSci.csv output/result_Turkish_BuiTecSci.csv output/result_Worldwide_BuiTecSci.csv > output/combined_BuiTecSci.csv
rm output/result_*
#python3 ../tools/filterCSV.py output/combined_BuiTecSci.csv split 20

python3 textAnalyzer.py ../../data/processed/France/Pol/ twitter Romance French Pol
python3 textAnalyzer.py ../../data/processed/Germany/Pol/ twitter Germanic German Pol
python3 textAnalyzer.py ../../data/processed/Greece/Pol/ twitter Greek Greek Pol
python3 textAnalyzer.py ../../data/processed/India/Pol/ twitter Indo-Aryan Indian Pol
python3 textAnalyzer.py ../../data/processed/Turkey/Pol/ twitter Turkic Turkish Pol
python3 textAnalyzer.py ../../data/processed/Native/Pol/ twitter Germanic English Pol
python3 textAnalyzer.py ../../data/processed/Worldwide/Pol/ twitter Worldwide Worldwide Pol
cat output/result_English_Pol.csv output/result_German_Pol.csv output/result_French_Pol.csv output/result_Greek_Pol.csv output/result_Indian_Pol.csv output/result_Turkish_Pol.csv output/result_Worldwide_Pol.csv > output/combined_Pol.csv
rm output/result_*
#python3 ../tools/filterCSV.py output/combined_Pol.csv split 20

cat output/combined_Pol.csv output/combined_SocSoc.csv output/combined_ArtCul.csv output/combined_BuiTecSci.csv > output/full_data_twitter.csv

python3 ../tools/filterCSV.py output/full_data_twitter.csv ngram 1000 twitter
rm ../../data/processed/ngrams/twitter/ngrams_twitter_*
mv output/ngrams* ../../data/processed/ngrams/twitter/

#python3 ../tools/filterCSV.py output/full_data_twitter.csv classifier




