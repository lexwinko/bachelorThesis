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
python3 ../tools/filterCSV.py output/combined_ArtCul.csv split 20