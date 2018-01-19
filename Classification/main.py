'''Main moudle  for Random Forest Algorithm

This module runs Random forest algorithm to
predict the count of bike rentals
'''

import os

import config
from pkg.analyze import (
    read_training_dataset_pandas,
    )


# using pandas
train_pandas=read_training_dataset_pandas(config.PATH_TO_TRAIN_FILE)
