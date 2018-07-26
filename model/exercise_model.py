
from base.base_model import BaseModel

import logging
import pdb

from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import SpatialDropout1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from keras.wrappers.scikit_learn import KerasRegressor


class ExerciseModel(BaseModel):
    def __init__(self, config=None):
        super()
        if config is None:
            config = json.load(io.open("configs/exercise_config.json"))

        self.config = config

        # Create model


        logging.info("Classifier created...")
        #clf = KerasRegressor(build_fn=self._create_model,
        #                     verbose=config["model"]["params"]["verbose"])
        clf = self._create_model()

        tokenizer = config["model"]["pipeline"]["tokenizer"]
        if tokenizer is None:
            tokenizer = Tokenizer()

        pipeline = Pipeline([
            # ('tokenizer', tokenizer), # FIXME tokenization and padding must
            # be done manually
            ('clf', clf)
        ])
        #self.pipeline = pipeline
        self.pipeline = clf

    def _create_model(self):
        config = self.config
        logging.info("Creating model layers from config file...")
        layers = []
        layers.append(Embedding(config["model"]["embedding"]["max_features"]+1,
                            output_dim=config["model"]["embedding"]["output_dim"],
                            input_length=config["model"]["embedding"]["input_length"], 
                            embeddings_initializer=config["model"]["embedding"]["embeddings_initializer"]))
        layers.append(SpatialDropout1D(rate=config["model"]["spacialdropout1d"]["rate"]))
        layers.append(GlobalMaxPooling1D())
        layers.append(Dense(units=config["model"]["dense"]["units"],
                         activation=config["model"]["dense"]["activation"]))
        adam = Adam(lr=config["model"]["adam"]["lr"],
                    beta_1=config["model"]["adam"]["beta_1"], 
                    beta_2=config["model"]["adam"]["beta_2"],
                    epsilon=config["model"]["adam"]["epsilon"],
                    decay=config["model"]["adam"]["decay"])

        logging.info("Adding layers to model...")
        model = Sequential()
        for layer in layers:
            model.add(layer)

        logging.info("Compiling the model...")
        model.compile(loss=config["model"]["compile"]["loss"],
                      optimizer=config["model"]["compile"]["optimizer"],
                      metrics=config["model"]["compile"]["metrics"])
        return model


    def fit(self, X, y):

        return self.pipeline.fit(X, y)


    def predict_proba(self, X):
        config = self.config
        tokenizer = config["model"]["pipeline"]["tokenizer"]
        label_encoder = config["model"]["label_encoder"]
        pdb.set_trace()
        # TODO debug results with this
        # z = ["I am doing sprint and jogging around the block"]
        # self.pipeline.predict(pad_sequences(tokenizer.texts_to_sequences(z), maxlen=config["model"]["embedding"]["input_length"]))
        # label_encoder.inverse_transform([1])

        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=config["model"]["embedding"]["input_length"])
        return self.pipeline.predict_proba(X)


    def load(self, loc=None):
        """Load this model from the file."""
        # FIXME does this work?
        if loc is None:
            loc = self.config["classifier"]
        self.model = joblib.load(loc)
        return self.model


    def save(self, loc=None):
        """Saves the model, if loc is None, it uses the location in config."""
        # FIXME does this work?
        if loc is None:
            loc = self.config["classifier"]
        logging.info("Saving  model to {}".format(self.config["classifier"]))
        return joblib.dump(self, loc)

