#! /usr/bin/python3
"""
This module when run, produces stats for classifiers.

Usage: time pipenv run python classifier_stats.py [--food] [--exercise] [--borne]
"""

import logging

import click

from data_loader.exercise_loader import ExerciseDataLoader
from data_loader.foodborne_loader import FoodBorneDataLoader
from data_loader.food_loader import FoodDataLoader
from trainer.exercise_trainer import ExerciseTrainer
from trainer.foodborne_trainer import FoodBorneTrainer
from trainer.food_trainer import FoodTrainer

from sklearn.metrics import precision_recall_fscore_support


@click.command()
@click.option('--borne', is_flag=True, help="The foodborne classifer statistics")
@click.option('--food', is_flag=True, help="The food classifer statistics")
@click.option('--exercise', is_flag=True, help="The exercise classifer statistics")
def main(borne, food, exercise):

    if borne:
        logging.info("Computing foodborne classifier stats...")
        #  Create data loader
        fbl = FoodBorneDataLoader()

        # Create data traininer
        fbt = FoodBorneTrainer()

        # Split training data
        (X_train, y_train) = fbl.get_train_data()
        logging.info(f"Training size: {len(X_train)}")

        (X_test, y_test) = fbl.get_test_data()
        logging.info(f"Testing size: {len(X_test)}")

        # TODO train
        fbt.train(X_train, y_train)

        y_pred = fbt.model.predict(X_test)
        results =  precision_recall_fscore_support(y_test, y_pred)
        data = {"precision": results[0],
                "recall": results[1],
                "fscore": results[2],
                "support": results[3],
                "train_size": len(y_train),
                "test_size": len(y_test)
               }



        # TODO Print training data size and split
        logging.info(data)
        # TODO evaluate results
        # TODO print result to stdout

    if food:
        logging.info("Computing food classifier stats...")
        # TODO Create data loader
        # TODO create data traininer
        # TODO Split training data
        # TODO train

        # TODO Print training data size and split
        # TODO evaluate results
        # TODO print result to stdout

    if exercise:
        logging.info("Computing exercise classifier stats...")
        # TODO Create data loader
        # TODO create data traininer
        # TODO Split training data
        # TODO train

        # TODO Print training data size and split
        # TODO evaluate results
        # TODO print result to stdout


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format='%(levelname)s %(asctime)s %(filename)s %(lineno)d: %(message)s')
    logging.info("logging enabled...")

    main()

