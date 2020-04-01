import pickle
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

'''
Description:
    Preprocesses and cleans the dataset of hotel reviews (and handwritten).

Authors:
    Yoshio Schermer (500760587)
'''

def fetch_reviews():
    # connecting to db
    engine = create_engine(
        'mysql+mysqlconnector://root:root@localhost/hotel_reviews')
    connection = engine.raw_connection()

    # define parameters to be passed in and out
    df = None

    start_query = time.time()
    try:
        cursor = connection.cursor()
        cursor.callproc("SelectReviews")

        # fetch result parameters
        for result in cursor.stored_results():
            reviews = result.fetchall()
            df = pd.DataFrame(reviews)

        cursor.close()
        connection.commit()
    finally:
        connection.close()
        end_query = time.time()
        print("Query took: " + str(end_query - start_query) + " seconds.\n")
    return df

def fetch_handwritten_reviews():
    # merging handwritten reviews
    filename = r"..\data_sources\handwritten_hotel_reviews.csv"
    handwritten_reviews = pd.read_csv(filename, sep=',')
    handwritten_reviews.columns = ['review', 'label']
    return handwritten_reviews

def clean():
    df = fetch_reviews()

    # naming columns
    df.columns = ["negative_review", "negative_word_count",
                  "positive_review", "positive_word_count"]

    # splitting dataframe into negative and positive
    # using np.split: 20+ seconds
    # using copy and drop: <1  second
    dfs = df.copy(deep=True)
    negative_reviews = df.drop(
        columns=['positive_review', "positive_word_count"])
    positive_reviews = dfs.drop(
        columns=['negative_review', "negative_word_count"])

    # renaming columns
    negative_reviews.columns = ['review', 'word_count']
    positive_reviews.columns = ['review', 'word_count']

    # remove bogus or empty reviews
    bogus_reviews = ["NOTHING", "Nothing", "nothing", "Nothing at all",
                     "No Negative", "No Positive", "n a", "N a", "N A", " "]
    negative_reviews = negative_reviews[negative_reviews.isin(
        {"review": bogus_reviews})['review'] == False]
    positive_reviews = positive_reviews[positive_reviews.isin(
        {"review": bogus_reviews})['review'] == False]

    # normalize negative reviews
    negative_threshold = 0
    negative_labels = (negative_reviews['word_count']
                       > negative_threshold).astype(int)
    negative_reviews['label'] = negative_labels

    # remove non-negative reviews
    negative_reviews = negative_reviews[negative_reviews['label'] == 1]

    # normalize positive reviews
    positive_threshold = 0
    positive_labels = (positive_reviews['word_count']
                       > positive_threshold).astype(int)
    positive_reviews['label'] = positive_labels

    # remove non-positive reviews
    positive_reviews = positive_reviews[positive_reviews['label'] == 1]

    # A negative review will have as label value: -1.
    # A positive review will have as label value: 1.
    negative_reviews = negative_reviews.assign(label=-1)

    # merging reviews and labels
    reviews = pd.concat([negative_reviews, positive_reviews],
                        join="inner", ignore_index=True)
    reviews = reviews.drop(columns=['word_count'])

    # merging handwritten reviews
    reviews = pd.concat([reviews, fetch_handwritten_reviews()],
                        join="inner", ignore_index=True)

    return reviews

def observe(reviews):
    # observe dataframe characteristics
    amount_negative_reviews = reviews[reviews['label'] == -1].shape[0]
    amount_positive_reviews = reviews[reviews['label'] == 1].shape[0]

    print("Amount of negative reviews: {}.".format(
        str(amount_negative_reviews)))
    print("Amount of positive reviews: {}.".format(
        str(amount_positive_reviews)))
    print("Predicting only 1 = {:.2f}% accuracy.\n".format(
        amount_positive_reviews / (amount_positive_reviews + amount_negative_reviews) * 100))
