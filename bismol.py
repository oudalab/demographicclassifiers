
"""
Usage: time python3 -m pipenv run python bismol.py
"""

import json
import logging
import pdb

import click
import pandas as pd

from data_loader.foodborne_loader import FoodBorneDataLoader
from data_loader.food_loader import FoodDataLoader
from trainer.foodborne_trainer import FoodBorneTrainer
from trainer.food_trainer import FoodTrainer


def foodborne_predict(data):
    """Takes in a twitter data frame that at least has 'id_str' and
    'status' column. Returns the data fram with an appended 'prediction'
    """

    # Create a pandas data frame with two columns
    #data = pd.DataFrame(data=([json.loads(t)['id_str'],
    #                    json.loads(t)['status']]  for t in tweets),
    #                    columns=['id_str', 'status'])

    # Train the foodborne predictor so we can use the
    # model to to the prediction
    logging.info("Load the foodborne data...")
    fbl = FoodBorneDataLoader()

    logging.info("Create the foodborne trainer...")
    fbt = FoodBorneTrainer()

    logging.info("Fit the foodborne model...")
    fbt.train(*fbl.get_train_data())

    logging.info("Running prediction...")
    result = fbt.model.predict(data["text"])
    # logging.info("{} <-- {}".format(result ))

    # Add results to data
    data = data.assign(prediction=result)

    return data


# TODO create interface for other food and exercise classifiers



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    z = pd.read_json("resources/tweet_exercise_20170202033134.json.gz", lines=True, precise_float=True)
    z['id_str'] = z['id_str'].apply(lambda x: "{:.0f}".format(x))
    z = z[['id_str', 'text']]
    z.fillna(value="", inplace=True)
    result = foodborne_predict(z)
    # pdb.set_trace()
    print(result[['id_str', 'text', 'prediction']])

    # Return non junk results
    print(result.query("prediction != 'junk'"))

    # TODO write out in a perticular file format

