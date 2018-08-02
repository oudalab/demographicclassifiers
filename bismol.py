
"""
Usage: time pipenv run python bismol.py <tweet-file> <output-file>
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


@click.command()
@click.option('--input', default="resources/tweet_exercise_20170202033134.json.gz")
@click.option('--output', default="")
@click.option('--borne', classifier, flag_value='borne', default=True)
@click.option('--food', classifier, flag_value='food')
@click.option('--exercise', classifier, flag_value='exercise')
def main(inputfile, outputfile, classifier):

    # Check if the file <p
    # z = pd.read_json("resources/tweet_exercise_20170202033134.json.gz", lines=True, precise_float=True)
    z = pd.read_json(inputfile, lines=True, precise_float=True)
    z['id_str'] = z['id_str'].apply(lambda x: "{:.0f}".format(x))
    z = z[['id_str', 'text']]
    z.fillna(value="", inplace=True)

    if classifier == 'borne':
        logging.info("Classifying {} using foodborne tweets.".format(inputfile))

        result = foodborne_predict(z)
        # pdb.set_trace()
        logging.info(result[['id_str', 'text', 'prediction']])

        # Return non junk results
        logging.info(result.query("prediction != 'junk'"))

        result = result.replace('\n',' ', regex=True)
        result.to_csv(outputfile, encoding="utf-8", sep="\t", index=False )
        
    elif classifier == 'exercise':
        logging.info("Classifying {} using exercise tweets.".format(inputfile))
        result = exercise_predict(z)
        result = result.replace('\n',' ', regex=True)
        result.to_csv(outputfile, encoding="utf-8", sep="\t", index=False )

    elif classifier == 'food':
        logging.info("Classifying {} using food tweets.".format(inputfile))
        result = food_predict(z)
        result = result.replace('\n',' ', regex=True)
        result.to_csv(outputfile, encoding="utf-8", sep="\t", index=False )

    else:
        logging.error("invalid flag selected")
        sys.exit("bad classifier flag")


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


def exercise_predict(data):
    """Takes in a twitter data frame that at least has 'id_str' and
    'status' column. Returns the data fram with an appended 'prediction'
    """
    logging.info("Load the exercise data...")
    el = ExerciseDataLoader()

    logging.info("Create the exercise trainer...")
    et = ExerciseTrainer()

    logging.info("Fit the exercise model...")
    et.train(*el.get_train_data())

    logging.info("Run prediction...")
    result = et.model.predict_proba(["text"])

    data = data.assign(prediction=result)

    return data


def food_prediction(data): 
    """Takes in a twitter data frame that at least has 'id_str' and
    'status' column. Returns the data fram with an appended 'prediction'
    """
    logging.info("Load the food data...")
    fl = FoodDataLoader()

    logging.info("Create the food trainer...")
    ft = FoodTrainer()

    logging.info("Fit the food model...")
    ft.train(*fl.get_train_data())

    logging.info("Running prediction...")
    result = ft.model.predict(["text"])

    data = data.assign(prediction=result)

    return data


# TODO create interface for other food and exercise classifiers


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format='%(levelname)s %(asctime)s %(filename)s %(lineno)d: %(message)s')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading the nltk tokenizer/punkt model")
        nltk.download('punkt')

    main()
    
