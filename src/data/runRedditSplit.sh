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
cat australia_00.csv ireland_00.csv newzealand_00.csv uk_00.csv us_00.csv > reddit_native.csv
cat australiaNE_00.csv irelandNE_00.csv newzealandNE_00.csv ukNE_00.csv usNE_00.csv > redditNE_native.csv
find . -type f ! -name "*reddit*" -exec rm -rf {} \;
python3 ../../../tools/filterCSV.py reddit_native.csv reformat reddit native
python3 ../../../tools/filterCSV.py reformat_reddit_native.csv tag Native English Native
python3 ../../../tools/filterCSV.py redditNE_native.csv reformat reddit nativeNE
python3 ../../../tools/filterCSV.py reformat_reddit_nativeNE.csv tag Native English Native

find . -type f ! -name "*tagged*" -exec rm -rf {} \;
#cat tagged_australia.csv tagged_ireland.csv tagged_newzealand.csv tagged_uk.csv tagged_us.csv > native.csv
#cat tagged_australiaNE.csv tagged_irelandNE.csv tagged_newzealandNE.csv tagged_ukNE.csv tagged_usNE.csv > nativeNE.csv
cd ../

#python3 ../../../tools/filterCSV.py australia_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py australiaNE_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py ireland_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py irelandNE_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py newzealand_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py newzealandNE_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py uk_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py ukNE_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py us_00.csv tag Native English Native
#python3 ../../../tools/filterCSV.py usNE_00.csv tag Native English Native
#find . -type f ! -name "*tagged*" -exec rm -rf {} \;
#cat tagged_australia.csv tagged_ireland.csv tagged_newzealand.csv tagged_uk.csv tagged_us.csv > native.csv
#cat tagged_australiaNE.csv tagged_irelandNE.csv tagged_newzealandNE.csv tagged_ukNE.csv tagged_usNE.csv > nativeNE.csv


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
python3 ../../../tools/filterCSV.py bulgaria_00.csv reformat reddit bulgaria
python3 ../../../tools/filterCSV.py bulgariaNE_00.csv reformat reddit bulgariaNE
python3 ../../../tools/filterCSV.py croatia_00.csv reformat reddit croatia
python3 ../../../tools/filterCSV.py croatiaNE_00.csv reformat reddit croatiaNE
python3 ../../../tools/filterCSV.py czech_00.csv reformat reddit czech
python3 ../../../tools/filterCSV.py czechNE_00.csv reformat reddit czechNE
python3 ../../../tools/filterCSV.py lithuania_00.csv reformat reddit lithuania
python3 ../../../tools/filterCSV.py lithuaniaNE_00.csv reformat reddit lithuaniaNE
python3 ../../../tools/filterCSV.py poland_00.csv reformat reddit poland
python3 ../../../tools/filterCSV.py polandNE_00.csv reformat reddit polandNE
python3 ../../../tools/filterCSV.py russia_00.csv reformat reddit russia
python3 ../../../tools/filterCSV.py russiaNE_00.csv reformat reddit russiaNE
python3 ../../../tools/filterCSV.py serbia_00.csv reformat reddit serbia
python3 ../../../tools/filterCSV.py serbiaNE_00.csv reformat reddit serbiaNE
python3 ../../../tools/filterCSV.py slovenia_00.csv reformat reddit slovenia
python3 ../../../tools/filterCSV.py sloveniaNE_00.csv reformat reddit sloveniaNE

python3 ../../../tools/filterCSV.py reformat_reddit_bulgaria.csv tag Balto-Slavic Bulgarian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_bulgariaNE.csv tag Balto-Slavic Bulgarian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_croatia.csv tag Balto-Slavic Croatian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_croatiaNE.csv tag Balto-Slavic Croatian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_czech.csv tag Balto-Slavic Czech NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_czechNE.csv tag Balto-Slavic Czech NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_lithuania.csv tag Balto-Slavic Lithuanian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_lithuaniaNE.csv tag Balto-Slavic Lithuanian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_poland.csv tag Balto-Slavic Polish NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_polandNE.csv tag Balto-Slavic Polish NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_russia.csv tag Balto-Slavic Russian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_russiaNE.csv tag Balto-Slavic Russian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_serbia.csv tag Balto-Slavic Serbian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_serbiaNE.csv tag Balto-Slavic Serbian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_slovenia.csv tag Balto-Slavic Slovene NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_sloveniaNE.csv tag Balto-Slavic Slovene NonNative
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_reformat_reddit_bulgaria.csv tagged_reformat_reddit_croatia.csv tagged_reformat_reddit_czech.csv tagged_reformat_reddit_lithuania.csv tagged_reformat_reddit_poland.csv tagged_reformat_reddit_russia.csv tagged_reformat_reddit_serbia.csv tagged_reformat_reddit_slovenia.csv > balto-slavic.csv
cat tagged_reformat_reddit_bulgariaNE.csv tagged_reformat_reddit_croatiaNE.csv tagged_reformat_reddit_czechNE.csv tagged_reformat_reddit_lithuaniaNE.csv tagged_reformat_reddit_polandNE.csv tagged_reformat_reddit_russiaNE.csv tagged_reformat_reddit_serbiaNE.csv tagged_reformat_reddit_sloveniaNE.csv > balto-slavicNE.csv
cd ../

#germanic 10000/1650
mkdir germanic
cd germanic/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Austria.tok.clean.csv | split -dl 825 --additional-suffix=.csv - austria_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Austria.tok.cleanNE.csv | split -dl 825 --additional-suffix=.csv - austriaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Finland.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - finland_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Finland.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - finlandNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Germany.tok.clean.csv | split -dl 825 --additional-suffix=.csv - germany_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Germany.tok.cleanNE.csv | split -dl 825 --additional-suffix=.csv - germanyNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Netherlands.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - netherlands_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Netherlands.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - netherlandsNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Norway.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - norway_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Norway.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - norwayNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Sweden.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - sweden_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Germanic/reddit.Sweden.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - swedenNE_

cat austria_00.csv germany_00.csv > germany.csv
cat austriaNE_00.csv germanyNE_00.csv > germanyNE.csv

python3 ../../../tools/filterCSV.py finland_00.csv reformat reddit finland
python3 ../../../tools/filterCSV.py finlandNE_00.csv reformat reddit finlandNE
python3 ../../../tools/filterCSV.py germany.csv reformat reddit germany
python3 ../../../tools/filterCSV.py germanyNE.csv reformat reddit germanyNE
python3 ../../../tools/filterCSV.py netherlands_00.csv reformat reddit netherlands
python3 ../../../tools/filterCSV.py netherlandsNE_00.csv reformat reddit netherlandsNE
python3 ../../../tools/filterCSV.py norway_00.csv reformat reddit norway
python3 ../../../tools/filterCSV.py norwayNE_00.csv reformat reddit norwayNE
python3 ../../../tools/filterCSV.py sweden_00.csv reformat reddit sweden
python3 ../../../tools/filterCSV.py swedenNE_00.csv reformat reddit swedenNE

python3 ../../../tools/filterCSV.py reformat_reddit_finland.csv tag Germanic Finnish NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_finlandNE.csv tag Germanic Finnish NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_germany.csv tag Germanic German NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_germanyNE.csv tag Germanic German NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_netherlands.csv tag Germanic Dutch NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_netherlandsNE.csv tag Germanic Dutch NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_norway.csv tag Germanic Norwegian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_norwayNE.csv tag Germanic Norwegian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_sweden.csv tag Germanic Swedish NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_swedenNE.csv tag Germanic Swedish NonNative


find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_reformat_reddit_finland.csv tagged_reformat_reddit_germany.csv tagged_reformat_reddit_netherlands.csv tagged_reformat_reddit_norway.csv tagged_reformat_reddit_sweden.csv > germanic.csv
cat tagged_reformat_reddit_finlandNE.csv tagged_reformat_reddit_germanyNE.csv tagged_reformat_reddit_netherlandsNE.csv tagged_reformat_reddit_norwayNE.csv tagged_reformat_reddit_swedenNE.csv > germanicNE.csv
cd ../

#greek 10000/10000
#mkdir greek
#cd greek/
#head -100000 /media/sf_Shared/reddit_filtered/nonnative/Greek/reddit.Greece.tok.clean.csv | split -dl 10000 --additional-suffix=.csv - greece_
#head -100000 /media/sf_Shared/reddit_filtered/nonnative/Greek/reddit.Greece.tok.cleanNE.csv | split -dl 10000 --additional-suffix=.csv - greeceNE_
#python3 ../../../tools/filterCSV.py greece_00.csv tag Greek Greek
#python3 ../../../tools/filterCSV.py greeceNE_00.csv tag Greek Greek
#find . -type f ! -name "*tagged*" -exec rm -rf {} \;
#cat tagged_greece.csv > greek.csv
#cat tagged_greeceNE.csv > greekNE.csv
#cd ../

mkdir greek
cd greek/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Greek/reddit.Greece.tok.clean.csv | split -dl 10000 --additional-suffix=.csv - greece_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Greek/reddit.Greece.tok.cleanNE.csv | split -dl 10000 --additional-suffix=.csv - greeceNE_

python3 ../../../tools/filterCSV.py greece_00.csv reformat reddit greece
python3 ../../../tools/filterCSV.py greeceNE_00.csv reformat reddit greeceNE
python3 ../../../tools/filterCSV.py reformat_reddit_greece.csv tag Greek Greek NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_greeceNE.csv tag Greek Greek NonNative
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_reformat_reddit_greece.csv > greek.csv
cat tagged_reformat_reddit_greece.csv > greekNE.csv
cd ../

#romance 10000/1650
mkdir romance
cd romance/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.France.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - france_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.France.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - franceNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Italy.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - italy_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Italy.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - italyNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Mexico.tok.clean.csv | split -dl 825 --additional-suffix=.csv - mexico_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Mexico.tok.cleanNE.csv | split -dl 825 --additional-suffix=.csv - mexicoNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Portugal.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - portugal_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Portugal.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - portugalNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Romania.tok.clean.csv | split -dl 1650 --additional-suffix=.csv - romania_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Romania.tok.cleanNE.csv | split -dl 1650 --additional-suffix=.csv - romaniaNE_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Spain.tok.clean.csv | split -dl 825 --additional-suffix=.csv - spain_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Romance/reddit.Spain.tok.cleanNE.csv | split -dl 825 --additional-suffix=.csv - spainNE_

cat mexico_00.csv spain_00.csv > spain.csv
cat mexicoNE_00.csv spainNE_00.csv > spainNE.csv

python3 ../../../tools/filterCSV.py france_00.csv reformat reddit france
python3 ../../../tools/filterCSV.py franceNE_00.csv reformat reddit franceNE
python3 ../../../tools/filterCSV.py italy_00.csv reformat reddit italy
python3 ../../../tools/filterCSV.py italyNE_00.csv reformat reddit italyNE
python3 ../../../tools/filterCSV.py portugal_00.csv reformat reddit portugal
python3 ../../../tools/filterCSV.py portugalNE_00.csv reformat reddit portugalNE
python3 ../../../tools/filterCSV.py romania_00.csv reformat reddit romania
python3 ../../../tools/filterCSV.py romaniaNE_00.csv reformat reddit romaniaNE
python3 ../../../tools/filterCSV.py spain.csv reformat reddit spain
python3 ../../../tools/filterCSV.py spainNE.csv reformat reddit spainNE

python3 ../../../tools/filterCSV.py reformat_reddit_france.csv tag Romance French NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_franceNE.csv tag Romance French NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_italy.csv tag Romance Italian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_italyNE.csv tag Romance Italian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_portugal.csv tag Romance Portugese NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_portugalNE.csv tag Romance Portugese NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_romania.csv tag Romance Romanian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_romaniaNE.csv tag Romance Romanian NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_spain.csv tag Romance Spanish NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_spainNE.csv tag Romance Spanish NonNative
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_reformat_reddit_france.csv tagged_reformat_reddit_italy.csv tagged_reformat_reddit_portugal.csv tagged_reformat_reddit_romania.csv tagged_reformat_reddit_spain.csv > romance.csv
cat tagged_reformat_reddit_franceNE.csv tagged_reformat_reddit_italyNE.csv tagged_reformat_reddit_portugalNE.csv tagged_reformat_reddit_romaniaNE.csv tagged_reformat_reddit_spainNE.csv > romanceNE.csv
cd ../

#turkic 10000/10000
mkdir turkic
cd turkic/
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Turkic/reddit.Turkey.tok.clean.csv | split -dl 10000 --additional-suffix=.csv - turkey_
head -100000 /media/sf_Shared/reddit_filtered/nonnative/Turkic/reddit.Turkey.tok.cleanNE.csv | split -dl 10000 --additional-suffix=.csv - turkeyNE_

python3 ../../../tools/filterCSV.py turkey_00.csv reformat reddit turkey
python3 ../../../tools/filterCSV.py turkeyNE_00.csv reformat reddit turkeyNE
python3 ../../../tools/filterCSV.py reformat_reddit_turkey.csv tag Turkic Turkish NonNative
python3 ../../../tools/filterCSV.py reformat_reddit_turkeyNE.csv tag Turkic Turkish NonNative
find . -type f ! -name "*tagged*" -exec rm -rf {} \;
cat tagged_reformat_reddit_turkey.csv > turkic.csv
cat tagged_reformat_reddit_turkey.csv > turkicNE.csv
cd ../

#uralic 10000/5000
# mkdir uralic
# cd uralic/
# head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Estonia.tok.clean.csv | split -dl 5000 --additional-suffix=.csv - estonia_
# head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Estonia.tok.cleanNE.csv | split -dl 5000 --additional-suffix=.csv - estoniaNE_
# head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Hungary.tok.clean.csv | split -dl 5000 --additional-suffix=.csv - hungary_
# head -100000 /media/sf_Shared/reddit_filtered/nonnative/Uralic/reddit.Hungary.tok.cleanNE.csv | split -dl 5000 --additional-suffix=.csv - hungaryNE_
# python3 ../../../tools/filterCSV.py estonia_00.csv tag Uralic Estonian
# python3 ../../../tools/filterCSV.py estoniaNE_00.csv tag Uralic Estonian
# python3 ../../../tools/filterCSV.py hungary_00.csv tag Uralic Hungarian
# python3 ../../../tools/filterCSV.py hungaryNE_00.csv tag Uralic Hungarian
# find . -type f ! -name "*tagged*" -exec rm -rf {} \;
# cat tagged_estonia.csv tagged_hungary.csv > uralic.csv
# cat tagged_estoniaNE.csv tagged_hungaryNE.csv > uralicNE.csv
# cd ../

cat native/tagged_reformat_reddit_native.csv balto-slavic/balto-slavic.csv germanic/germanic.csv romance/romance.csv > full_reddit_data.csv
cat native/tagged_reformat_reddit_native.csv native/tagged_reformat_reddit_nativeNE.csv > full_reddit_native.csv
cat balto-slavic/balto-slavic.csv germanic/germanic.csv romance/romance.csv balto-slavic/balto-slavicNE.csv germanic/germanicNE.csv romance/romanceNE.csv > full_reddit_nonnative.csv
cat native/tagged_reformat_reddit_nativeNE.csv balto-slavic/balto-slavicNE.csv germanic/germanicNE.csv romance/romanceNE.csv> full_reddit_dataNE.csv
cat full_reddit_data.csv full_reddit_dataNE.csv > full_combined_reddit_data.csv
find . -type f ! -name "full*" -exec rm -rf {} \;

#mv * ../../../data/external/

mkdir native
cd native/
cp ../../../../data/raw/Native/ArtCul/* .
cp ../../../../data/raw/Native/BuiTecSci/* .
cp ../../../../data/raw/Native/Pol/* .
cp ../../../../data/raw/Native/SocSoc/* .
cat * > twitter_native.csv
python3 ../../../tools/filterCSV.py twitter_native.csv reformat twitter native
find . -type f ! -name "*reformat*" -exec rm -rf {} \;
python3 ../../../tools/filterCSV.py reformat_twitter_native.csv tag Native English Native
cd ../


mkdir indo-european
cd indo-european/
cp ../../../../data/raw/France/ArtCul/* .
cp ../../../../data/raw/France/BuiTecSci/* .
cp ../../../../data/raw/France/Pol/* .
cp ../../../../data/raw/France/SocSoc/* .
cat * > twitter_france.csv
mkdir france
python3 ../../../tools/filterCSV.py twitter_france.csv reformat twitter france
python3 ../../../tools/filterCSV.py reformat_twitter_france.csv tag Indo-European French NonNative
mv tagged_* france
find . -type f ! -name "*tagged*" -exec rm -rf {} \;

cp ../../../../data/raw/Germany/ArtCul/* .
cp ../../../../data/raw/Germany/BuiTecSci/* .
cp ../../../../data/raw/Germany/Pol/* .
cp ../../../../data/raw/Germany/SocSoc/* .
cat * > twitter_germany.csv
mkdir germany
python3 ../../../tools/filterCSV.py twitter_germany.csv reformat twitter germany
python3 ../../../tools/filterCSV.py reformat_twitter_germany.csv tag Indo-European German NonNative
mv tagged_* germany
find . -type f ! -name "*tagged*" -exec rm -rf {} \;

cp ../../../../data/raw/Russia/ArtCul/* .
cat * > twitter_russia.csv
mkdir russia
python3 ../../../tools/filterCSV.py twitter_russia.csv reformat twitter russia
python3 ../../../tools/filterCSV.py reformat_twitter_russia.csv tag Indo-European Russian NonNative
mv tagged_* russia
find . -type f ! -name "*tagged*" -exec rm -rf {} \;

cp ../../../../data/raw/Greece/ArtCul/* .
cp ../../../../data/raw/Greece/BuiTecSci/* .
cp ../../../../data/raw/Greece/Pol/* .
cp ../../../../data/raw/Greece/SocSoc/* .
cat * > twitter_greece.csv
mkdir greece
python3 ../../../tools/filterCSV.py twitter_greece.csv reformat twitter greece
python3 ../../../tools/filterCSV.py reformat_twitter_greece.csv tag Indo-European Greek NonNative
mv tagged_* greece
find . -type f ! -name "*tagged*" -exec rm -rf {} \;

cat france/* germany/* russia/* greece/* > twitter_indo-european.csv
cd ../

mkdir indo-aryan
cd indo-aryan
cp ../../../../data/raw/India/ArtCul/* .
cp ../../../../data/raw/India/BuiTecSci/* .
cp ../../../../data/raw/India/Pol/* .
cp ../../../../data/raw/India/SocSoc/* .
cat * > twitter_indo-aryan.csv
python3 ../../../tools/filterCSV.py twitter_indo-aryan.csv reformat twitter indo-aryan
python3 ../../../tools/filterCSV.py reformat_twitter_indo-aryan.csv tag Indo-Aryan Indian NonNative
cd ../

mkdir turkic
cd turkic/
cp ../../../../data/raw/Turkey/ArtCul/* .
cp ../../../../data/raw/Turkey/BuiTecSci/* .
cp ../../../../data/raw/Turkey/Pol/* .
cp ../../../../data/raw/Turkey/SocSoc/* .
cat * > twitter_turkic.csv
python3 ../../../tools/filterCSV.py twitter_turkic.csv reformat twitter turkic
python3 ../../../tools/filterCSV.py reformat_twitter_turkic.csv tag Turkic Turkish NonNative
cd ../

mkdir japonic
cd japonic/
cp ../../../../data/raw/Japan/ArtCul/* .
cat * > twitter_japonic.csv
python3 ../../../tools/filterCSV.py twitter_japonic.csv reformat twitter japonic
python3 ../../../tools/filterCSV.py reformat_twitter_japonic.csv tag Japonic Japanese NonNative
cd ../

