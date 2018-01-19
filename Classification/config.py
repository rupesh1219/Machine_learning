'''
configurations for the project
'''

import os

# path variables
PATH_TO_MAIN = os.path.dirname(os.path.abspath(__file__))
PATH_TO_INSTANCE = os.path.join(PATH_TO_MAIN, 'instance')
PATH_TO_TEST_FILE = os.path.join(PATH_TO_INSTANCE, 'data/test.csv')
PATH_TO_TRAIN_FILE = os.path.join(PATH_TO_INSTANCE, 'data/train.csv')
PATH_TO_PLOTS = os.path.join(PATH_TO_INSTANCE, 'plots/')

# file variables
RANDOM_FOREST_FILE = 'PREDICTED_COUNT_RANDOM_FOREST.csv'
DECISION_TREE_FILE = 'PREDICTED_COUNT_DECISION_TREE.csv'
