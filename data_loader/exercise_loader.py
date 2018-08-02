
from base.base_data_loader import BaseDataLoader

import io
import json
import logging
import pdb
import re

import pandas as pd

from sklearn.model_selection import train_test_split

class ExerciseDataLoader(BaseDataLoader):
    def __init__(self, config=None):
        # load the config
        logging.info("__init__Loade Reading the config file.")
        if config is None:
            config = json.load(io.open("configs/exercise_config.json"))

        self.config = config

        # note classified_tweets_latest.json has a different syntax
        # Format that thy data file first

        logging.info("Reading and formatting classified_tweets_latest")
        etweets = pd.read_json("resources/classified_tweets_latest.json.gz",
                               dtype=object,
                               orient="columns")
        logging.info("Classified tweets loaded")

        # Add the tweet text as a column
        etweets = etweets.assign(tweets_text = etweets["tweet"]
                                        .apply(lambda tweet : eval(tweet))
                                        .apply(lambda tweet : tweet["text"]))
 
        # Tweets that are not foodborne or are not labeled
        etweets = etweets.loc[etweets["exercise"] != "None"]
        etweets = etweets.loc[etweets["label_data"] != '']
        etweets = etweets.loc[etweets.exercise == '1'] # Foodborne specific

        # Make labels all lower case
        etweets['label_data'] = etweets.label_data.str.lower()

        # Tokenize the tweet on spaces
        # Remove all non word or non-number characters
        clean_text = r"([!@#$]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
        clean_f = lambda x: ' '.join(re.sub(clean_text,'',x).split())
        etweets["tweets_text"] = etweets.tweets_text.apply(clean_f)

        # Now, Read other files
        logging.info("Reading and cleaning files: {}".format(config["training_files"][1:]))
        twts = pd.concat([pd.read_json(gzfile, dtype=object, orient="columns", compression="gzip")
                   for gzfile in config["training_files"][1:]],
                   ignore_index=True)

        twts = twts.assign(tweets_text = twts.tweet
                                    .apply(lambda tweet: eval(tweet))
                                    .apply(lambda text: text['text']))
        twts.exercise = twts.exercise.astype("category")

        twts = twts.loc[twts["exercise"] != "None"]
        twts = twts.loc[twts["label_data"] != '']
        twts = twts.loc[twts.exercise == '1']
        twts["label_data"] = twts.label_data.str.lower()
        twts["tweets_text"] = twts.tweets_text.apply(clean_f)

        logging.info("Combining files")
        data = pd.concat([ etweets[['label_data','tweets_text']],
                            twts[['label_data','tweets_text']] ])

        data = data.loc[data.label_data.isin(config["labels"])]

        logging.info("Assigning labels to {}".format(config["labels"]))
        self.labels = config["labels"]

        logging.info("Now splitting data into test and train sets")
        # self.data = data
        tts = train_test_split(data.tweets_text.str.lower(),
                                       data.label_data,
                                       test_size=0.2)

        (self.X_train, self.X_test, self.y_train, self.y_test) = tts


    def get_train_data(self):
        return self.X_train, self.y_train


    def get_test_data(self):
        return self.X_test, self.y_test


    def filter_tweet(self):
        # TODO check the wordlist and return true if it contains one of the
        # words and satisfies all criteria

        # There are three types of checks.
        # 1) Exact string match
        # 2) Exact Phrase match
        # 3) String match with negative words.
        # Combine them all into one big regex?
        pass

