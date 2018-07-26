
"""
Usage: time python3 -m pipenv run python bismol.py arg1 arg2
"""
import sys
import json
import logging
import pdb

import click
import pandas as pd
import nltk
nltk.download('punkt')

from data_loader.foodborne_loader import FoodBorneDataLoader
from data_loader.food_loader import FoodDataLoader
from trainer.foodborne_trainer import FoodBorneTrainer
from trainer.food_trainer import FoodTrainer

print(sys.argv)

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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

# TODO create interface for other food and exercise classifiers


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    z = pd.read_json(sys.argv[1], lines=True, dtype=object, precise_float=True)
    if len(intersection(['id_str'], list(z))) == 0:
         z['id_str'] = z['_id']
    # z['id_str'] = z['id_str'].apply(lambda x: "{:.0f}".format(x))
    z = z[['id_str', 'text']]
    z.fillna(value="", inplace=True)
    result = foodborne_predict(z)
    # pdb.set_trace()
    print(result[['id_str', 'text', 'prediction']])

    # Return non junk results
    print(result.query("prediction != 'junk'"))
    result.to_csv(sys.argv[2], encoding="utf-8")
    
    # TODO write out in a perticular file format
