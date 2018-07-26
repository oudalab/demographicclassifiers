#!/bin/bash

# Usage: bash comparefiles

# Create a new output file of the following form.
# Compares the ids from the original file and the new file
# return the differing ids.
# pipenv run python bismol_nlc.py tweets_foodborne_geotagged_00.json testout-`date '+%Y-%m-%d-%H-%M-%S'`.csv


echoerr () { echo "$@" 1>&2; }

comparefiles () {

	local outfile
	local jsonfile

	local jsonfile="$1"
	local outfile="$2"

	echoerr "outfile: $outfile"
	echoerr "jsonfile: $jsonfile"

	sleep 5

	diff <(cat "$jsonfile" | jq -r "[._id, .text] | @tsv" | awk '{print $1}') \
		<(cat "$outfile" | awk '{print $1}')

}
comparefiles tweets_foodborne_geotagged_00.json $(ls -tn testout-* | awk '{print $NF}')
