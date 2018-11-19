#!/bin/bash
set -e
# requires environment variable SINGLESTATION to be set

. ${HOME}/.profile 2>&1 > /dev/null

EXEC=${HOME}/locator/main.py
CONFIG=${SINGLESTATION}/bin/marsgui_locate.cfg

LOG_IN=${SINGLESTATION}/log/bodywavedistance_in.yml
LOG_OUT=${SINGLESTATION}/log/bodywavedistance_out.yml

STDIN=$(cat)


echo -e "$STDIN" > $LOG_IN 
python $EXEC $LOG_IN $LOG_OUT --plot 
cat $LOG_OUT

exit 1
