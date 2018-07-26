
from base.base_trainer import BaseTrain
from model.exercise_model import ExerciseModel

import io
import json
import logging
import pdb

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class ExerciseTrainer(BaseTrain):
    def __init__(self, model=None, config=None):
        logging.info("__init__ trainer Reading the config file.")
        if config is None:
            config = json.load(io.open("configs/exercise_config.json"))

        self.config = config


    def train(self, X, y):
        """Fits and returns the classifier pipeline.
        If `cache` is True, it looks for the saved model and saves it if it
        doesn not exist.
        """
        logging.info("Training the food model.")
        config = self.config

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        config["model"]["label_encoder"] = label_encoder
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        labels = onehot_encoder.fit_transform(integer_encoded)
        
        max_length = np.max([len(i.split()) for i in X]) 
        config["model"]["embedding"]["input_length"] = max_length

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(X))
        max_features = len(tokenizer.word_docs.keys())
        config["model"]["embedding"]["max_features"] = max_features

        # Add tokenizer to the pipeline
        config["model"]["pipeline"]["tokenizer"] = tokenizer
        
        # update properties of the model
        # Update the config file so we can instantiate the proper model
        if not hasattr(self, "model"):
            self.model = ExerciseModel(config=self.config)

        # self.model.compile() # Not needed

        # FIXME <-- need to add the below to the pipeline
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=config["model"]["embedding"]["input_length"])

        logging.info("training the model...")
        self.model.fit(X,
                       integer_encoded,
                       # batch_size=config["model"]["fit"]["batch_size"],
                       # epochs=config["model"]["fit"]["epochs"],
                       # validation_split=config["model"]["fit"]["validation_split"]
                        ) 
        logging.info("Done training.")
        return self.model



