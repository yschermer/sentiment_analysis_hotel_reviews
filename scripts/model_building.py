import pickle
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from data_cleaning import clean

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn

'''
Description:
    First, preprocesses and cleans the dataset of hotel reviews.
    Subsequently, building models with the cleaned dataset.
    Finally, saving the models as pickle objects.

Authors:
    Yoshio Schermer (500760587)
'''

reviews = clean()

# split into train and test set
X = reviews["review"]
y = reviews["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = [
    # {
    #     "id": "mnb",
    #     "name": "Multinomial Naive Bayes",
    #     "object": MultinomialNB(),
    #     "parameters": {
    #             'vect__max_features': (1000, 10000),
    #             'vect__stop_words': ["english", None],
    #             'vect__ngram_range': [(1, 1), (1, 2)],
    #             'tfidf__use_idf': (True, False),
    #     }
    # },
    # {
    #     "id": "bnb",
    #     "name": "Bernoulli Naive Bayes",
    #     "object": BernoulliNB(),
    #     "parameters": {
    #         'vect__max_features': (1000, 10000),
    #         'vect__stop_words': ["english", None],
    #         'vect__ngram_range': [(1, 1), (1, 2)],
    #         'tfidf__use_idf': (True, False),
    #     }
    # },
    # {
    #     "id": "lsvc",
    #     "name": "Linear Support Vector Classifier",
    #     "object": LinearSVC(),
    #     "parameters": {
    #         'vect__max_features': (1000, 10000),
    #         'vect__stop_words': ["english", None],
    #         'vect__ngram_range': [(1, 1), (1, 2)],
    #         'tfidf__use_idf': (True, False),
    #         'clf__C': [0.1, 1],
    #     }
    # },
    # {
    #     "id": "lr",
    #     "name": "Logistic Regression",
    #     "object": LogisticRegression(solver="lbfgs"),
    #     "parameters": {
    #         'vect__max_features': (1000, 10000),
    #         'vect__stop_words': ["english", None],
    #         'vect__ngram_range': [(1, 1), (1, 2)],
    #         'tfidf__use_idf': (True, False),
    #         'clf__max_iter': [10000]
    #     }
    # },
]

confusion_matrices = []

for model in models:
    # setting up a pipeline for text feature extraction
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()), ('clf', model['object'])])

    gs_clf = GridSearchCV(
        text_clf, model['parameters'], scoring='accuracy', cv=5, n_jobs=1)

    print("Performing grid search cross validation for {}...".format(
        model['name']))
    gs_clf.fit(X_train, y_train)
    print("Completed grid search cross validation for {}.\n".format(
        model['name']))

    y_pred = gs_clf.predict(X_test)

    # evaluate classifier performance
    print(
        "{} PERFORMANCE EVALUATION\n-------------------".format(model['name'].upper()))
    print("Mean cross-validated score of best estimator: {}.".format(gs_clf.best_score_))
    print("Parameter setting that gave the best results: {}.".format(
        gs_clf.best_params_))
    print("Mean fit time of best estimator: {} seconds.".format(
        gs_clf.cv_results_['mean_fit_time'][gs_clf.best_index_]))
    print("Classification report:\n{}".format(
        classification_report(y_test, y_pred, digits=4)))
    print("Confusion Matrix:\n{}\n".format(
        confusion_matrix(y_test, y_pred, [-1, 1])))

    confusion_matrices.append({
        "name": model['name'],
        "confusion_matrix": pd.DataFrame(confusion_matrix(y_test, y_pred, [-1, 1]), [-1, 1], [-1, 1])
    })

    # save model
    print("Saving classifier...")
    filename = '../classifiers/{}.sav'.format(model['id'])
    pickle.dump(gs_clf, open(filename, 'wb'))
    print("Classifier successfully saved in: classifiers/{}.sav\n".format(model['id']))

for confusion_matrix in confusion_matrices:
    plt.figure(figsize=(10, 7))
    plt.title(confusion_matrix['name'], fontsize =20)
    sn.set(font_scale=1.4)  # for label size
    ax = sn.heatmap(confusion_matrix['confusion_matrix'], annot=True,
                    annot_kws={"size": 16}, fmt='g')  # font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()