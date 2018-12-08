[![Build Status](https://travis-ci.org/sstaehler/locator.svg?branch=master)](https://travis-ci.org/sstaehler/locator)
--
Locator
--
Tries to locate earthquakes based on picked arrival times and a set of velocity models in MQS-format. 

- 
Testing:
-
The package contains a script to create test files, based on picked surface
wave arrivals from a 3D simulation.
```bash
# Set necessary environment variable with path to model files
export SINGLESTATION=./locator/data/models/

# 8 model files are included in this repository.
# To download more, use 
wget -q https://polybox.ethz.ch/index.php/s/ibufo2dYhyUfu6j/download -O /tmp/250models.tar.bz2
tar -xf /tmp/250models.tar.bz2 -C locator/data/models/data/bodywave/

# Create a test file with travel times
# First argument selects event, second selects event depth
python ./locator/data/tests/create_test.py 41 70 

# Run the locator code
python ./main.py ./locator_input.yml locator_output_test.yml --plot
```
Check the created png files in the rundir


For further information, contact the author or the MarsQuakeService.