
from base.base_data_loader import BaseDataLoader

import io
import json
import logging
import re

import pandas as pd

from sklearn.model_selection import train_test_split

class FoodBorneDataLoader(BaseDataLoader):
    def __init__(self, config=None):
        # load the config
        logging.info("__init__Loade Reading the config file.")
        if config is None:
            config = json.load(io.open("configs/foodborne_config.json"))

        self.config = config

        # note classified_tweets_latest.json has a different syntax
        # Format that thy data file first

        logging.info("Reading and formatting classified_tweets_latest")
        fbtweets = pd.read_json("resources/classified_tweets_latest.json.gz",
                                orient="columns",
                                compression="gzip",
                                precise_float=True)

        # Add the tweet text as a column
        fbtweets = fbtweets.assign(tweets_text = fbtweets["tweet"]
                                        .apply(lambda tweet : eval(tweet))
                                        .apply(lambda tweet : tweet["text"]))

        # Tweets that are not foodborne or are not labeled
        fbtweets = fbtweets.loc[fbtweets["food"] != "None"] # FIXME verify that this is not a bug, why food and bot borne here?
        fbtweets = fbtweets.loc[fbtweets["label_data"] != '']
        fbtweets = fbtweets.loc[fbtweets.borne == '1'] # Foodborne specific

        # Make labels all lower case
        fbtweets['label_data'] = fbtweets.label_data.str.lower()

        # Tokenize the tweet on spaces
        # Remove all non word or non-number characters
        clean_text = r"([!@#$]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
        clean_f = lambda x: ' '.join(re.sub(clean_text,'',x).split())
        fbtweets["tweets_text"] = fbtweets.tweets_text.apply(clean_f)

        # Now, Read other files ####
        logging.info("Reading and cleaning files: {}".format(config["training_files"][1:]))
        twts = pd.concat([pd.read_json(gzfile, orient="columns", compression="gzip", precise_float=True)
                   for gzfile in config["training_files"][1:]],
                   ignore_index=True)

        twts = twts.assign(tweets_text = twts.tweet
                                    .apply(lambda tweet: eval(tweet))
                                    .apply(lambda text: text['text']))
        twts.borne = twts.borne.astype("category")

        twts = twts.loc[twts["borne"] != "None"]
        twts = twts.loc[twts["label_data"] != '']
        twts = twts.loc[twts.borne == '1']
        twts["label_data"] = twts.label_data.str.lower()
        twts["tweets_text"] = twts.tweets_text.apply(clean_f)
        twts = twts.loc[twts.borne == 1] # Foodborne specific # FIXME this twice?

        logging.info("Combining files")
        data = pd.concat([ fbtweets[['label_data','tweets_text']],
                            twts[['label_data','tweets_text']] ])

        data = data.loc[data.label_data.isin(config["labels"])]

        logging.info("Assigning labels to {}".format(config["labels"]))
        self.labels = config["labels"]

        logging.info("Changing the irrelevant tag to junk")
        irrelevant2junk = lambda x : "junk" if x == "irrelevant" else x
        data.label_data = data.label_data.apply(irrelevant2junk)



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
