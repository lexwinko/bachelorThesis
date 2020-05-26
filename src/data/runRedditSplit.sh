#!/bin/sh
mkdir split 
cd split/
#native 10000/2000
mkdir native
cd native/
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.Australia.tok.clean.csv | split -dl 2000 --additional-suffix=.csv - australia_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.Australia.tok.cleanNE.csv | split -dl 2000 --additional-suffix=.csv - australiaNE_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.Ireland.tok.clean.csv | split -dl 2000 --additional-suffix=.csv - ireland_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.Ireland.tok.cleanNE.csv | split -dl 2000 --additional-suffix=.csv - irelandNE_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.NewZealand.tok.clean.csv | split -dl 2000 --additional-suffix=.csv - newzealand_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.NewZealand.tok.cleanNE.csv | split -dl 2000 --additional-suffix=.csv - newzealandNE_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.UK.tok.clean.csv | split -dl 2000 --additional-suffix=.csv - uk_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.UK.tok.cleanNE.csv | split -dl 2000 --additional-suffix=.csv - ukNE_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.US.tok.clean.csv | split -dl 2000 --additional-suffix=.csv - us_
head -100000 /media/sf_Shared/reddit_filtered/native/reddit.US.tok.cleanNE.csv | split -dl 2000 --additional-suffix=.csv - usNE_
python3 ../../../tools/filterCSV.py australia_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py australiaNE_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py ireland_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py irelandNE_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py newzealand_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py newzealandNE_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py uk_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py ukNE_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py us_00.csv tag Romance Native
python3 ../../../tools/filterCSV.py usNE_00.csv tag Romance Native
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_australia.csv tagged_ireland.csv tagged_newzealand.csv tagged_uk.csv tagged_us.csv > native.csv
cat tagged_australiaNE.csv tagged_irelandNE.csv tagged_newzealandNE.csv tagged_ukNE.csv tagged_usNE.csv > nativeNE.csv
cd ../


#balto-slavic 10000/1250
mkdir balto-slavic
cd balto-slavic/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Bulgaria.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - bulgaria_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Bulgaria.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - bulgariaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Croatia.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - croatia_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Croatia.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - croatiaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Czech.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - czech_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Czech.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - czechNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Lithuania.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - lithuania_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Lithuania.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - lithuaniaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Poland.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - poland_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Poland.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - polandNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Russia.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - russia_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Russia.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - russiaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Serbia.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - serbia_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Serbia.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - serbiaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Slovenia.tok.clean.csv | split -dl 1250 --additional-suffix=.csv - slovenia_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Balto-Slavic/reddit.Slovenia.tok.cleanNE.csv | split -dl 1250 --additional-suffix=.csv - sloveniaNE_
python3 ../../../tools/filterCSV.py bulgaria_00.csv tag Balto-Slavic Bulgarian
python3 ../../../tools/filterCSV.py bulgariaNE_00.csv tag Balto-Slavic Bulgarian
python3 ../../../tools/filterCSV.py croatia_00.csv tag Balto-Slavic Croatian
python3 ../../../tools/filterCSV.py croatiaNE_00.csv tag Balto-Slavic Croatian
python3 ../../../tools/filterCSV.py czech_00.csv tag Balto-Slavic Czech
python3 ../../../tools/filterCSV.py czechNE_00.csv tag Balto-Slavic Czech
python3 ../../../tools/filterCSV.py lithuania_00.csv tag Balto-Slavic Lithuanian
python3 ../../../tools/filterCSV.py lithuaniaNE_00.csv tag Balto-Slavic Lithuanian
python3 ../../../tools/filterCSV.py poland_00.csv tag Balto-Slavic Polish
python3 ../../../tools/filterCSV.py polandNE_00.csv tag Balto-Slavic Polish
python3 ../../../tools/filterCSV.py russia_00.csv tag Balto-Slavic Russian
python3 ../../../tools/filterCSV.py russiaNE_00.csv tag Balto-Slavic Russian
python3 ../../../tools/filterCSV.py serbia_00.csv tag Balto-Slavic Serbian
python3 ../../../tools/filterCSV.py serbiaNE_00.csv tag Balto-Slavic Serbian
python3 ../../../tools/filterCSV.py slovenia_00.csv tag Balto-Slavic Slovene
python3 ../../../tools/filterCSV.py sloveniaNE_00.csv tag Balto-Slavic Slovene
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_bulgaria.csv tagged_croatia.csv tagged_czech.csv tagged_lithuania.csv tagged_poland.csv tagged_russia.csv tagged_serbia.csv tagged_slovenia.csv > balto-slavic.csv
cat tagged_bulgariaNE.csv tagged_croatiaNE.csv tagged_czechNE.csv tagged_lithuaniaNE.csv tagged_polandNE.csv tagged_russiaNE.csv tagged_serbiaNE.csv tagged_sloveniaNE.csv > balto-slavicNE.csv
cd ../

#germanic 10000/1650
mkdir germanic
cd germanic/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Austria.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - austria_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Austria.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - austriaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Finland.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - finland_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Finland.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - finlandNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Germany.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - germany_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Germany.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - germanyNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Netherlands.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - netherlands_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Netherlands.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - netherlandsNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Norway.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - norway_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Norway.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - norwayNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Sweden.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - sweden_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Sweden.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - swedenNE_
python3 ../../../tools/filterCSV.py austria_00.csv tag Germanic German
python3 ../../../tools/filterCSV.py austriaNE_00.csv tag Germanic German
python3 ../../../tools/filterCSV.py finland_00.csv tag Germanic Finnish
python3 ../../../tools/filterCSV.py finlandNE_00.csv tag Germanic Finnish
python3 ../../../tools/filterCSV.py germany_00.csv tag Germanic German
python3 ../../../tools/filterCSV.py germanyNE_00.csv tag Germanic German
python3 ../../../tools/filterCSV.py netherlands_00.csv tag Germanic Dutch
python3 ../../../tools/filterCSV.py netherlandsNE_00.csv tag Germanic Dutch
python3 ../../../tools/filterCSV.py norway_00.csv tag Germanic Norwegian
python3 ../../../tools/filterCSV.py norwayNE_00.csv tag Germanic Norwegian
python3 ../../../tools/filterCSV.py sweden_00.csv tag Germanic Swedish
python3 ../../../tools/filterCSV.py swedenNE_00.csv tag Germanic Swedish
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_austria.csv tagged_finland.csv tagged_germany.csv tagged_netherlands.csv tagged_norway.csv tagged_sweden.csv > germanic.csv
cat tagged_austriaNE.csv tagged_finlandNE.csv tagged_germanyNE.csv tagged_netherlandsNE.csv tagged_norwayNE.csv tagged_swedenNE.csv > germanicNE.csv
cd ../

#greek 10000/10000
mkdir greek
cd greek/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Greek/reddit.Greece.tok.clean.csv | split -dl 10000 --additional-suffix=.csv - greece_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Greek/reddit.Greece.tok.cleanNE.csv | split -dl 10000 --additional-suffix=.csv - greeceNE_
python3 ../../../tools/filterCSV.py greece_00.csv tag Greek Greek
python3 ../../../tools/filterCSV.py greeceNE_00.csv tag Greek Greek
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_greece.csv > greek.csv
cat tagged_greeceNE.csv > greekNE.csv
cd ../

#romance 10000/1650
mkdir romance
cd romance/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.France.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - france_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.France.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - franceNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Italy.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - italy_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Italy.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - italyNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Mexico.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - mexico_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Mexico.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - mexicoNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Portugal.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - portgual_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Portugal.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - portugalNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Romania.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - romania_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Romania.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - romaniaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Spain.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - spain_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Spain.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - spainNE_
python3 ../../../tools/filterCSV.py france_00.csv tag Romance French
python3 ../../../tools/filterCSV.py franceNE_00.csv tag Romance French
python3 ../../../tools/filterCSV.py italy_00.csv tag Romance Italian
python3 ../../../tools/filterCSV.py italyNE_00.csv tag Romance Italian
python3 ../../../tools/filterCSV.py mexico_00.csv tag Romance Spanish
python3 ../../../tools/filterCSV.py mexicoNE_00.csv tag Romance Spanish
python3 ../../../tools/filterCSV.py portgual_00.csv tag Romance Portugese
python3 ../../../tools/filterCSV.py portugalNE_00.csv tag Romance Portugese
python3 ../../../tools/filterCSV.py romania_00.csv tag Romance Romanian
python3 ../../../tools/filterCSV.py romaniaNE_00.csv tag Romance Romanian
python3 ../../../tools/filterCSV.py spain_00.csv tag Romance Spanish
python3 ../../../tools/filterCSV.py spainNE_00.csv tag Romance Spanish
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_france.csv tagged_italy.csv tagged_mexico.csv tagged_portugal.csv tagged_romania.csv tagged_spain.csv > romance.csv
cat tagged_franceNE.csv tagged_italyNE.csv tagged_mexicoNE.csv tagged_portugalNE.csv tagged_romaniaNE.csv tagged_spainNE.csv > romanceNE.csv
cd ../

#turkic 10000/10000
mkdir turkic
cd turkic/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Turkic/reddit.Turkey.tok.clean.csv | split -dl 10000 --additional-suffix=.csv - turkey_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Turkic/reddit.Turkey.tok.cleanNE.csv | split -dl 10000 --additional-suffix=.csv - turkeyNE_
python3 ../../../tools/filterCSV.py turkey_00.csv tag Turkic Turkish
python3 ../../../tools/filterCSV.py turkeyNE_00.csv tag Turkic Turkish
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_turkey.csv > turkic.csv
cat tagged_turkeyNE.csv > turkicNE.csv
cd ../

#uralic 10000/5000
mkdir uralic
cd uralic/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Estonia.tok.clean.csv | split -dl 5000 --additional-suffix=.csv - estonia_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Estonia.tok.cleanNE.csv | split -dl 5000 --additional-suffix=.csv - estoniaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Hungary.tok.clean.csv | split -dl 5000 --additional-suffix=.csv - hungary_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Hungary.tok.cleanNE.csv | split -dl 5000 --additional-suffix=.csv - hungaryNE_
python3 ../../../tools/filterCSV.py estonia_00.csv tag Uralic Estonian
python3 ../../../tools/filterCSV.py estoniaNE_00.csv tag Uralic Estonian
python3 ../../../tools/filterCSV.py hungary_00.csv tag Uralic Hungarian
python3 ../../../tools/filterCSV.py hungaryNE_00.csv tag Uralic Hungarian
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_estonia.csv tagged_hungary.csv > uralic.csv
cat tagged_estoniaNE.csv tagged_hungaryNE.csv > uralicNE.csv
cd ../

cat native/native.csv balto-slavic/balto-slavic.csv germanic/germanic.csv greek/greek.csv romance/romance.csv turkic/turkic.csv uralic/uralic.csv > reddit_data.csv
cat native/nativeNE.csv balto-slavic/balto-slavicNE.csv germanic/germanicNE.csv greek/greekNE.csv romance/romanceNE.csv turkic/turkicNE.csv uralic/uralicNE.csv > reddit_dataNE.csv
cat reddit_data.csv reddit_dataNE.csv > combined_reddit_data.csv

mv * ../../../data/external/



