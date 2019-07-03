#!/bin/bash
set -e

# requires environment variable SIMONSLOCATOR to be set

. ${HOME}/.profile 2>&1 > /dev/null


EXEC=${SIMONSLOCATOR}/main.py

LOG_IN=${HOME}/bodywavedistance_in.yml
LOG_OUT=${HOME}/bodywavedistance_out.yml

STDIN=$(cat)


echo -e "$STDIN" > $LOG_IN 
python $EXEC $LOG_IN $LOG_OUT --plot -d 100 --model_output
cat $LOG_OUT

export otime=$(awk '/origin_time_sum/{print $NF}' $LOG_OUT  | sed 's/://g')
mkdir -p event_$otime
xdg-open depth_distance.png & 

cp depth_distance.png event_$otime/depth_distance_$otime.png
tar -cf models_used_$otime.tar models_location 
gzip -f models_used_$otime.tar

exit 1
