from __future__ import division
import nltk
from nltk.tokenize import TweetTokenizer
import random
import pandas as pd
import pickle
import re, string, unicodedata
import inflect
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# you need this function bc
# joblib cannot mix things from python 2 and python 3
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def determine_most_likely(all_intents_text_df, given_intent, end_date, how_many):
    # determine the how_many most likely intents to follow the given_intent
    # using data from the beginning of 2016 to end_date

    all_intents_text_df["Asked_Date_Time"] = pd.to_datetime(all_intents_text_df["Asked_Date_Time"], 
                                                            format="%Y-%m-%d %H:%M:%S")
    # sort date
    all_intents_text_df = all_intents_text_df.sort_values(by=['Asked_Date_Time'])

    mask = (all_intents_text_df['Asked_Date_Time'] <= end_date)
    before_texts = all_intents_text_df.loc[mask]
    # sort by websession id and then by entry order
    sorted_before_texts = before_texts.sort_values(['WebSessionID', "EntryOrder"], ascending=[True, True])
    intents = list(sorted_before_texts["Intent"])
    # treat each intent as a word
    intent_dict = {} # key = intent with spaces removed, value = original intent
    for intent in intents:
        word_intent = intent.replace(" ","")
        intent_dict[word_intent] = intent
    intents = [intent.replace(" ","") for intent in intents]
    cfreq_intents_2gram = nltk.ConditionalFreqDist(nltk.bigrams(intents))
    # here are the how_many most frequent intents to come after the given intent, with their frequencies
    given_intent = given_intent.replace(" ", "")
    most_likely_to_follow = cfreq_intents_2gram[given_intent].most_common(how_many)
    most_likely_to_follow = [intent_dict[intent[0]] for intent in most_likely_to_follow]
    return most_likely_to_follow


# preprocessing: https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def get_word_list(texts, prob, fpml_preprocess=False):
    # http://www.katrinerk.com/courses/python-worksheets/language-models-in-python
    # cprob_2gram["Mileage"].prob("Plan")
    tknzr = TweetTokenizer()
    words = []
    # randomly choose strings to add
    # total = len(texts)
    # count = 1
    for string in texts:
        # print(count, " out of ", total)
        # print(string)
        if random.random() < prob:
            # print(tknzr.tokenize(str(string)))
            if fpml_preprocess:
                word_list = tknzr.tokenize(str(string))
            else:
                word_list = normalize(tknzr.tokenize(str(string)))
            # print(word_list)
            words += word_list
        # count += 1
    return words