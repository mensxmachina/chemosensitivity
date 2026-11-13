# Py
* filter.py:  Main filtering script
* utils.py:   Various functions & classes imported in filter.py


# Folders
## data
Necessary files for filter.py to run
* outcomes.csv
* outcomes_bool.csv
* outcomes_metadata.csv

## precomp
Folder containing intermediate calculations for fast loading
* tgf-init.pcl:  an initialized object with aggregation pre-calulated.
* filter_max1_N10_FC10_nnan10_CO25-75>10.pcl:  The final instance used in publication.

## targets
Folder containing the finalized target to be run
* TARGETS-BINNED-averaged.csv:  Averaged compounds + binned to 3 categories (sensitive, average, resistant)
* TARGETS-GI50-averaged.csv:    Averaged GI50 values
