
import logging

import click

from data_loader.exercise_loader import ExerciseDataLoader
from data_loader.foodborne_loader import FoodBorneDataLoader
from data_loader.food_loader import FoodDataLoader
from trainer.exercise_trainer import ExerciseTrainer
from trainer.foodborne_trainer import FoodBorneTrainer
from trainer.food_trainer import FoodTrainer


@click.command()
@click.option('--text',
              default="I had tacos they give me gas and make me vomit!",
              help='This is the text from a tweet.')
def main(text):

    exercise(text)
    food(text)
    foodborne(text)


def exercise(text):
    logging.info("Load the exercise data...")
    el = ExerciseDataLoader()

    logging.info("Create the exercise trainer...")
    et = ExerciseTrainer()

    logging.info("Fit the exercise model...")
    et.train(*el.get_train_data())

    logging.info("Run prediction...")
    result = et.model.predict_proba([text])
    logging.info("{} <-- {}".format(result, text))



def foodborne(text):
    logging.info("Load the foodborne data...")
    fbl = FoodBorneDataLoader()

    logging.info("Create the foodborne trainer...")
    fbt = FoodBorneTrainer()

    logging.info("Fit the foodborne model...")
    fbt.train(*fbl.get_train_data())

    logging.info("Running prediction...")
    result = fbt.model.predict([text])
    logging.info("{} <-- {}".format(result, text))

    fbt.save()

    return result


def food(text):
    logging.info("Load the food data...")
    fl = FoodDataLoader()

    logging.info("Create the food trainer...")
    ft = FoodTrainer()

    logging.info("Fit the food model...")
    ft.train(*fl.get_train_data())

    logging.info("Running prediction...")
    result = ft.model.predict([text])
    logging.info("{} <-- {}".format(result, text))

    ft.save()

    return result


def read_tweet_file(filename):
    """Takes in a tweet file .json.gz. Prints out a file with [id_str, prediction, conf]
    """

    pass



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    main()
