
from base.base_trainer import BaseTrain

import io
import json
import logging

import nltk
from nltk import word_tokenize

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class FoodBorneTrainer(BaseTrain):
    def __init__(self, model=None, config=None):
        logging.info("__init__ trainer Reading the config file.")
        if config is None:
            config = json.load(io.open("configs/foodborne_config.json"))

        self.config = config

        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=word_tokenize,
                                     stop_words="english",
                                     max_df=0.5,
                                     min_df=2,
                                     ngram_range=(1,3),
                                     max_features=None)
        svc = SVC(kernel='rbf',
                  decision_function_shape='ovo',
                  C=10,
                  gamma = 1,
                  probability=True)
        p = Pipeline([("Foodborne Count Vectorizer", vectorizer),
                      ("Foodborne Tfidf Transformer", TfidfTransformer()),
                      ("Foodborne Support Vector Classifier", svc)])

        self.model = p



    def train(self, X, y):
        """Fits and returns the classifier pipeline.
        If `cache` is True, it looks for the saved model and saves it if it
        doesn ot exist.
        """

        logging.info("Training the foodborne model...")
        self.model.fit(X, y)
        logging.info("Done training.")
        return self.model


    def load(self, loc=None):
        """Load this model from the file."""
        if loc is None:
            loc = self.config["classifier"]
        self.model = joblib.load(loc)
        return self.model


    def save(self, loc=None):
        """Saves the model, if loc is None, it uses the location in config."""
        if loc is None:
            loc = self.config["classifier"]
        logging.info("Saving  model to {}".format(self.config["classifier"]))
        return joblib.dump(self, loc)

