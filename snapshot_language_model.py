from __future__ import division
from spelling import correction
from autocorrect import spell
import nltk
from nltk.tokenize import TweetTokenizer
import pandas as pd
from preprocess_slm import *
import numpy as np
import time
from dateutil.relativedelta import *
from datetime import datetime
from fbprophet import Prophet
import matplotlib.pyplot as plt

def slm(texts, prob, fpml_preprocess=False):

    words = get_word_list(texts, prob, fpml_preprocess)


    # bigram calculator
    cfreq_2gram = nltk.ConditionalFreqDist(nltk.bigrams(words))
    cprob_2gram = nltk.ConditionalProbDist(cfreq_2gram, nltk.MLEProbDist)

    # unigram calculator
    # unigram_prob("airline")
    len_corpus = len(words)
    freq_1gram = nltk.FreqDist(words)
    def unigram_prob(word):
        return freq_1gram[word] / len_corpus

    return cprob_2gram, unigram_prob, len_corpus

# https://nlp.stanford.edu/pubs/linguistic_change_lifecycle.pdf
# the lower this is , the less surprising 
# the higher this is, the more surprising
def determine_cross_entropy_bigrams(string,cprob_2gram, normalize_len, unigram_prob, unknown_unigram_probability, verbose=True):
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(string)
    # normalize
    tokens = tokens[0:normalize_len]
    bigram_list = list(nltk.bigrams(tokens))
    the_sum = 0
    for bigram in bigram_list:
        first_word = bigram[0]
        second_word = bigram[1]
        if verbose:
            print(first_word, second_word)

        a = unigram_prob(first_word)
        b = cprob_2gram[first_word].prob(second_word)
        
        if a == 0:
            a = unknown_unigram_probability
            
            
        if b == 0:
            # print("0")
            b = unknown_unigram_probability
            

        prob = a*b

        the_sum += np.log(prob)
    if len(bigram_list) == 0:
        # print(string)
        raise ValueError("String must contain at least two tokens")
    return -(1/len(bigram_list)) * the_sum

def sliding_language_detector(given_intent, train_size, thresh=100000000000):

    all_intents_text_df=load_pickle("all_intents_text_df_python_2")

    mask = (all_intents_text_df['Intent'] == given_intent)
    slm_convs = all_intents_text_df.loc[mask]
    slm_convs = slm_convs.drop_duplicates(subset=["WebSessionID", "EntryOrder"])
    slm_convs["Asked_Date_Time"] = pd.to_datetime(slm_convs["Asked_Date_Time"], format="%Y-%m-%d %H:%M:%S")
    slm_convs = slm_convs.sort_values(by=['Asked_Date_Time'])
    # create training set
    before_texts = slm_convs[0:train_size]
    train_texts = before_texts["Input Converted"].values
    # create testing set
    test_texts = slm_convs[train_size:]



    tknzr = TweetTokenizer()

    my_dict = {}
    count = train_size
    while count < len(slm_convs):

        # train a bigram and unigram slm on the training test
        cprob_2gram, unigram_prob, len_corpus = slm(train_texts, 1, True)

        test_set = []
        ces = []
        perplexities = []
        web_session_ids = []

        final_test = len(test_texts)

        for index, row in test_texts.iterrows():
            print("Index: ", index)
            text = row["Input Converted"]
            print("Text: ", text)
            date = row["Asked_Date_Time"]
            print("Date: ", date)
            web_session_id = row["WebSessionID"]
            print("WebSessionID: ", web_session_id)
            # preprocess
            tokens = normalize(tknzr.tokenize(text))
            bigram_list = list(nltk.bigrams(tokens))
            if len(bigram_list) > 0:
                test_set.append(text)
                unknown_unigram_probability = 1 / len_corpus
                ce = determine_cross_entropy_bigrams(string=text, cprob_2gram=cprob_2gram, 
                                                     normalize_len=30, unigram_prob=unigram_prob,
                                                     unknown_unigram_probability=unknown_unigram_probability,
                                                     verbose=False)
                ces.append(ce)

                web_session_ids.append(web_session_id)

            if len(test_set) > 1:
                # http://www.cs.yale.edu/homes/radev/nlpclass/slides2017/213.pdf
                # https://stats.stackexchange.com/questions/285798/perplexity-and-cross-entropy-for-n-gram-models
                the_perplex = np.log(ce)
                perplexities.append(the_perplex)
                print("Perplexity: ", the_perplex)
                # we should consider perplexity average
                # https://en.wikipedia.org/wiki/Perplexity

                # if np.mean(perplexities) > thresh:
                if (the_perplex > thresh) or (the_perplex == 0):
                    print("Time to slide the SLM")
                    print("\n\n\n")
                    my_dict[date] = {"CES": ces, "Texts": test_set, "WebSessionIDs": web_session_ids}
                    break

            count += 1

        # update train texts
        before_texts = slm_convs[count-train_size:count]
        train_texts = before_texts["Input Converted"].values

        # update test texts
        test_texts = slm_convs[count:]

    return my_dict

def get_outliers(my_dict, all_intents_text_df):

    websessionids = []
    convs = []
    turn_predicted = []

    all_ces = []
    all_texts = []
    for i in sorted(my_dict.keys()) :  
        print(i)
        all_ces += my_dict[i]["CES"][:-1]
        all_texts += my_dict[i]["Texts"][:-1]
        # plt.plot(my_dict[i]["CES"])
        

        fb_prophet_model = Prophet(
                               growth='linear',
                               yearly_seasonality=False,
                               weekly_seasonality=False,
                               daily_seasonality=False,
                               interval_width=.999)
        # make up some dates
        dates = [str(item) for item in list(pd.date_range(start='2018-01-01', end='2018-12-31', periods=len(my_dict[i]["CES"])))]
        fb_df_train = pd.DataFrame({"ds": dates, 
                                    "y": my_dict[i]["CES"]})
        periods = 0
        fb_prophet_model.fit(fb_df_train, verbose=False)
        future = fb_prophet_model.make_future_dataframe(periods=periods)
        fcst = fb_prophet_model.predict(future)
        indices = []
        for k in range(len(fcst)):
            if fcst["yhat_upper"][k] < my_dict[i]["CES"][k]:
                indices.append(k)
        # print(indices)
        ys = [my_dict[i]["CES"][j] for j in indices]
        # print(ys)
        # plt.scatter(indices, ys,color="r", marker='x')


        if indices:
            
            for q in indices:
                print(my_dict[i]["WebSessionIDs"][q])
                print(my_dict[i]["Texts"][q])

                if my_dict[i]["WebSessionIDs"][q] == "DIFFERENT INTENT":
                    
                    if "xx" not in my_dict[i]["Texts"][q]:

                        websessionids.append("")
                        convs.append("")
                        websessionids.append("DIFFERENT INTENT")
                        convs.append(my_dict[i]["Texts"][q])

                # no trickery
                else:
                    mask = (all_intents_text_df['WebSessionID']==my_dict[i]["WebSessionIDs"][q])
                    test = all_intents_text_df.loc[mask]
                    test = test.sort_values(by=['Asked_Date_Time'])
                    print(test["Input"].values)

                    if my_dict[i]["WebSessionIDs"][q] not in websessionids:

                        if "xx" not in my_dict[i]["Texts"][q]:

                            websessionids.append("")
                            convs.append("")
                            turn_predicted.append("")

                            convs += list(test["Input"].values)
                            websessionids += len(test["Input"].values) * [my_dict[i]["WebSessionIDs"][q]]

                            turn_predicted_value = [""] * len(test["Input"].values)
                            turn_predicted_value[0] = my_dict[i]["Texts"][q]
                            turn_predicted += turn_predicted_value
            
        # plt.show()
        
        print("\n\n---------------------------------\n\n")

    tagging_df = pd.DataFrame({"WebSessionID": websessionids, "Full Conversation": convs, "Preprocessed Turn Predicted to be Anomalous": turn_predicted})
    return tagging_df







def sliding_language_detector_2(given_intent, trick_intent, train_size, thresh=100000000000):

    all_intents_text_df=load_pickle("all_intents_text_df_python_2")

    mask = (all_intents_text_df['Intent'] == given_intent)
    slm_convs = all_intents_text_df.loc[mask]
    slm_convs = slm_convs.drop_duplicates(subset=["WebSessionID", "EntryOrder"])
    slm_convs["Asked_Date_Time"] = pd.to_datetime(slm_convs["Asked_Date_Time"], format="%Y-%m-%d %H:%M:%S")
    slm_convs = slm_convs.sort_values(by=['Asked_Date_Time'])
    # create training set
    before_texts = slm_convs[0:train_size]
    train_texts = before_texts["Input Converted"].values
    # create testing set
    test_texts = slm_convs[train_size:]


    # create trick set
    trick_mask = (all_intents_text_df['Intent'] == trick_intent)
    trick_convs = all_intents_text_df.loc[trick_mask]
    trick_convs = trick_convs.drop_duplicates(subset=["WebSessionID", "EntryOrder"])
    trick_convs["Asked_Date_Time"] = pd.to_datetime(trick_convs["Asked_Date_Time"], format="%Y-%m-%d %H:%M:%S")
    trick_convs = trick_convs.sort_values(by=['Asked_Date_Time'])
    trick_texts = trick_convs["Input Converted"].values

    tknzr = TweetTokenizer()

    my_dict = {}
    count = train_size
    different_intent_count = 0
    while count < len(slm_convs):

        # train a bigram and unigram slm on the training test
        cprob_2gram, unigram_prob, len_corpus = slm(train_texts, 1, True)

        test_set = []
        ces = []
        perplexities = []
        web_session_ids = []

        final_test = len(test_texts)

        for index, row in test_texts.iterrows():

            if random.random() > .9:
                print("--------------")
                print("Different Intent")
                text = trick_texts[random.randint(0,len(trick_texts))]
                print(text)
                web_session_id = "DIFFERENT INTENT"
                different_intent_count += 1
            else:
                print("Index: ", index)
                text = row["Input Converted"]
                print("Text: ", text)
                date = row["Asked_Date_Time"]
                print("Date: ", date)
                web_session_id = row["WebSessionID"]
                print("WebSessionID: ", web_session_id)
            # preprocess
            tokens = normalize(tknzr.tokenize(text))
            bigram_list = list(nltk.bigrams(tokens))
            if len(bigram_list) > 0:
                test_set.append(text)
                unknown_unigram_probability = 1 / len_corpus
                ce = determine_cross_entropy_bigrams(string=text, cprob_2gram=cprob_2gram, 
                                                     normalize_len=30, unigram_prob=unigram_prob,
                                                     unknown_unigram_probability=unknown_unigram_probability,
                                                     verbose=False)
                ces.append(ce)

                web_session_ids.append(web_session_id)

            if len(test_set) > 1:
                # http://www.cs.yale.edu/homes/radev/nlpclass/slides2017/213.pdf
                # https://stats.stackexchange.com/questions/285798/perplexity-and-cross-entropy-for-n-gram-models
                the_perplex = np.log(ce)
                perplexities.append(the_perplex)
                print("Perplexity: ", the_perplex)
                # we should consider perplexity average
                # https://en.wikipedia.org/wiki/Perplexity

                # if np.mean(perplexities) > thresh:
                if (the_perplex > thresh) or (the_perplex == 0):
                    print("Time to slide the SLM")
                    print("\n\n\n")
                    my_dict[date] = {"CES": ces, "Texts": test_set, "WebSessionIDs": web_session_ids}
                    break

            count += 1

        # update train texts
        before_texts = slm_convs[count-train_size:count]
        train_texts = before_texts["Input Converted"].values

        # update test texts
        test_texts = slm_convs[count:]
    print(different_intent_count)
    return my_dict


def sliding_language_detector_voc(texts, train_size, thresh=100000000000):


    # create training set
    train_texts = texts[0:train_size]
    # create testing set
    test_texts = texts[train_size:]



    tknzr = TweetTokenizer()

    my_dict = {}
    count = train_size
    while count < len(texts):

        # train a bigram and unigram slm on the training test
        cprob_2gram, unigram_prob, len_corpus = slm(train_texts, 1, True)

        test_set = []
        ces = []
        perplexities = []

        final_test = len(test_texts)

        for text in test_texts:

            text = str(text)
            print("Text: ", text)
            
            # preprocess
            tokens = normalize(tknzr.tokenize(text))
            bigram_list = list(nltk.bigrams(tokens))
            if len(bigram_list) > 0:
                test_set.append(text)
                unknown_unigram_probability = 1 / len_corpus
                ce = determine_cross_entropy_bigrams(string=text, cprob_2gram=cprob_2gram, 
                                                     normalize_len=30, unigram_prob=unigram_prob,
                                                     unknown_unigram_probability=unknown_unigram_probability,
                                                     verbose=False)
                ces.append(ce)

            if len(test_set) > 1:
                # http://www.cs.yale.edu/homes/radev/nlpclass/slides2017/213.pdf
                # https://stats.stackexchange.com/questions/285798/perplexity-and-cross-entropy-for-n-gram-models
                the_perplex = np.log(ce)
                perplexities.append(the_perplex)
                print("Perplexity: ", the_perplex)
                # we should consider perplexity average
                # https://en.wikipedia.org/wiki/Perplexity

                # if np.mean(perplexities) > thresh:
                if (the_perplex > thresh) or (the_perplex == 0):
                    print("Time to slide the SLM")
                    print("\n\n\n")
                    my_dict[count] = {"CES": ces, "Texts": test_set}
                    break

            count += 1

        # update train texts
        train_texts = texts[count-train_size:count]

        # update test texts
        test_texts = texts[count:]

    return my_dict

def get_outliers_VOC(my_dict):

    convs = []

    all_ces = []
    all_texts = []

    outlier_texts = [] 
    for i in sorted(my_dict.keys()) :  
        print(i)
        all_ces += my_dict[i]["CES"][:-1]
        all_texts += my_dict[i]["Texts"][:-1]
        # plt.plot(my_dict[i]["CES"])
        

        fb_prophet_model = Prophet(
                               growth='linear',
                               yearly_seasonality=False,
                               weekly_seasonality=False,
                               daily_seasonality=False,
                               interval_width=.9)
        # make up some dates
        dates = [str(item) for item in list(pd.date_range(start='2018-01-01', end='2018-12-31', periods=len(my_dict[i]["CES"])))]
        fb_df_train = pd.DataFrame({"ds": dates, 
                                    "y": my_dict[i]["CES"]})
        periods = 0
        fb_prophet_model.fit(fb_df_train, verbose=False)
        future = fb_prophet_model.make_future_dataframe(periods=periods)
        fcst = fb_prophet_model.predict(future)
        indices = []
        for k in range(len(fcst)):
            if fcst["yhat_upper"][k] < my_dict[i]["CES"][k]:
                indices.append(k)
        # print(indices)
        ys = [my_dict[i]["CES"][j] for j in indices]
        print(ys)
        # plt.scatter(indices, ys,color="r", marker='x')



        if indices:
               
            for q in indices:
                print(my_dict[i]["Texts"][q])
                outlier_texts.append(my_dict[i]["Texts"][q])




                
                
        
        # plt.show()
        
        # print("\n\n---------------------------------\n\n")

    tagging_df = pd.DataFrame({"Predicted Anomalies": outlier_texts})
    return tagging_df
