#!/bin/bash

module load lowcharts

FILENAME=$1
cat $FILENAME | cut -d$'\t' -f2 | sort | lowcharts common-terms
