#!/user/bin/bash

# This scripts run tests over the classifiers
# It also return stats for the classifiers


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
#testfoodborne

testfood () {

time pipenv run python bismol.py --food \
	resources/tweet_food_20170911104612.json.gz \
	testout-`date '+%Y-%m-%d-%H-%M-%S'`.csv
}
testfood

# -------------------- Stats scripts

statsexercise () {
	time pipenv run python classifier_stats.py --exercise
}
statsexercise

statsborne() {
	time pipenv run python classifier_stats.py --borne
}
statsborne

statsfood () {
	time pipenv run python classifier_stats.py --food
}
statsfood


