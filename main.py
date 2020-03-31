import sys
import pickle

'''
Description:
    Takes a review and returns the sentiment per classifier, which are:
    - Multinomial Naive Bayes
    - Bernoulli Naive Bayes
    - Linear Support Vector Classifier
    - Logistic Regression

    Usage: python main.py "Type review here..."

    Argument:
        Review (as string enclosed by quotation marks)

    Returns:
        Prints sentiment

Authors:
    Yoshio Schermer (500760587)
'''

review = sys.argv[1]

print("Given review is: ", review)

models = [
    {
        "id": "mnb",
        "name": "Multinomial Naive Bayes"
    },
    {
        "id": "bnb",
        "name": "Bernoulli Naive Bayes"
    },
    {
        "id": "lsvc",
        "name": "Linear Support Vector Classifier"
    },
    {
        "id": "lr",
        "name": "Logistic Regression"
    },
]

for model in models:
    clf = pickle.load(open("classifiers/" + model['id'] + ".sav", 'rb'))
    pred = clf.predict([review])
    print("The review is classified as: {} (using {})".format("negative" if pred == -1 else "positive", model['name']))
