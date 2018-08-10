#! /usr/bin/python3
"""
This module when run, produces stats for classifiers.

Usage: time pipenv run python classifier_stats.py [--food] [--exercise] [--borne]
"""

import json
import logging
import pdb

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

        # train
        fbt.train(X_train, y_train)

        y_pred = fbt.model.predict(X_test)
        results = precision_recall_fscore_support(y_test, y_pred)
        data = {"precision": results[0].tolist(),
                "recall": results[1].tolist(),
                "fscore": results[2].tolist(),
                "support": results[3].tolist(),
                "train_size": len(y_train),
                "test_size": len(y_test)
               }

        # Print training data size and split
        logging.info(json.dumps(data))

        # TODO print result to stdout

    if food:
        logging.info("Computing food classifier stats...")
        # Create data loader
        fl = FoodDataLoader()

        # Create data traininer
        ft = FoodTrainer()

        # Split training data
        (X_train, y_train) = fl.get_train_data()
        logging.info(f"Training size: {len(X_train)}")

        (X_test, y_test) = fl.get_test_data()
        logging.info(f"Testing size: {len(X_test)}")

        # Train
        ft.train(X_train, y_train)

        # Print training data size and split
        y_pred = ft.model.predict(X_test)
        results = precision_recall_fscore_support(y_test, y_pred)
        data = {"precision": results[0].tolist(),
                "recall": results[1].tolist(),
                "fscore": results[2].tolist(),
                "support": results[3].tolist(),
                "train_size": len(y_train),
                "test_size": len(y_test)
               }

        logging.info(json.dumps(data))

        # TODO print result to stdout

    if exercise:
        logging.info("Computing exercise classifier stats...")
        # Create data loader
        el = ExerciseDataLoader()

        # Create data traininer
        et = ExerciseTrainer()

        # Split training data
        (X_train, y_train) = el.get_train_data()
        logging.info(f"Training size: {len(X_train)}")

        (X_test, y_test) = el.get_test_data()
        logging.info(f"Testing size: {len(X_test)}")

        # Train
        et.train(X_train, y_train)
        pdb.set_trace()

        # Print training data size and split
        y_pred = et.model.predict_proba(X_test)
        results = precision_recall_fscore_support(y_test, y_pred)
        data = {"precision": results[0].tolist(),
                "recall": results[1].tolist(),
                "fscore": results[2].tolist(),
                "support": results[3].tolist(),
                "train_size": len(y_train),
                "test_size": len(y_test)
                }

        logging.info(json.dumps(data))

        # TODO print result to stdout


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format='%(levelname)s %(asctime)s %(filename)s %(lineno)d: %(message)s')
    logging.info("logging enabled...")

    main()

