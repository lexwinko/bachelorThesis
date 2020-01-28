# Predicting Foreign Users from English conversations on Social Media

## Prerequisites
> tested on ubuntu 18.04

> Python 3

> openjdk-8-jre

### textAnalyzer
[ark-tweet-nlp](http://www.cs.cmu.edu/~ark/TweetNLP/)

[Twokenize Python Wrapper](https://github.com/ianozsvald/ark-tweet-nlp-python/blob/master/CMUTweetTagger.py)

```
sudo apt install pkg-config libicu-dev
pip3 install pandas polyglot PyICU pycld2 morfessor python-Levenshtein nltk
```

### aspell
[aspell-python](https://github.com/WojciechMula/aspell-python)
```
sudo apt install libaspell-dev

# In git repository
python3 setup.3.py build
sudo python3 setup.3.py install
```

### language-check
[language-check](https://github.com/myint/language-check)
```
# In git repository
python3 setup.py build
sudo python3 setup.py install
```

### textClassifier

```
pip3 install tensorflow sklearn
```
