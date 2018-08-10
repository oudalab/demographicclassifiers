#! /usr/bin/python3
"""
This module when run, produces stats for classifiers.

Usage: time pipenv run python classifier_stats.py [--food] [--exercise] [--borne]
"""

import logging

import click


@click.command()
@click.option('--borne', is_flag=True, help="The foodborne classifer statistics")
@click.option('--food', is_flag=True, help="The food classifer statistics")
@click.option('--exercise', is_flag=True, help="The exercise classifer statistics")
def main(borne, food, exercise):

    if borne:
        logging.info("Computing foodborne classifier stats")

    if food:
        logging.info("Computing food classifier stats")

    if exercise:
        logging.info("Computing exercise classifier stats")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format='%(levelname)s %(asctime)s %(filename)s %(lineno)d: %(message)s')
    logging.info("logging enabled...")

    main()

