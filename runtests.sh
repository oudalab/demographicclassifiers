#!/user/bin/bash

# This scripts run tests over the classifiers


testexercise () {

time pipenv run python bismol.py --exercise \
	resources/tweet_exercise_20170202033134.json.gz \
	testout-`date '+%Y-%m-%d-%H-%M-%S'`.csv

}
#testexercise


testfoodborne () {

time pipenv run python bismol.py --borne \
	resources/tweets_foodborne_geotagged_00.json \
	testout-`date '+%Y-%m-%d-%H-%M-%S'`.csv
}
testfoodborne
