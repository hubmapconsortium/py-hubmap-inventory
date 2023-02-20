#!/bin/bash

TYPE=$1
FILENAME=$2

if [[ "$TYPE" == "extension" ]]; then
	cat $FILENAME | cut -d$'\t' -f2 | sort | lowcharts common-terms
fi

if [[ "$TYPE" == "filetype" ]]; then
	cat $FILENAME | cut -d$'\t' -f4 | sort | lowcharts common-terms
fi

if [[ "$TYPE" == "mime-type" ]]; then
	cat $FILENAME | cut -d$'\t' -f7 | sort | lowcharts common-terms
fi
