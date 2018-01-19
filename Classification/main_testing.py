"""
OBJECTIVE: predict the total count of rented bikes (=demand)
Independent Variables
---------------------
datetime:   date and hour in "mm/dd/yyyy hh:mm" format
season:     Four categories-> 1 = spring, 2 = summer, 3 = fall, 4 = winter
holiday:    whether the day is a holiday or not (1/0)
workingday: whether the day is neither a weekend nor holiday (1/0)
weather:    Four Categories of weather
            1-> Clear, Few clouds, Partly cloudy, Partly cloudy
            2-> Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
            3-> Light Snow and Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
            4-> Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
temp:       hourly temperature in Celsius
atemp:      "feels like" temperature in Celsius
humidity:   relative humidity
windspeed:  wind speed
Dependent Variables
-------------------
registered: number of registered user
casual:     number of non-registered user
count:      number of total rentals (registered + casual)
"""

import datetime as dt
import os
import csv

from config import (
    PATH_TO_TRAIN_FILE,
    PATH_TO_TEST_FILE,
    PATH_TO_PLOTS,
    RANDOM_FOREST_FILE,
    DECISION_TREE_FILE,
)
from pkg.primitives import PandasDataFrame

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


###############################################################################
# Exploring dataset and analyzing data
###############################################################################

train_dataframe = pd.read_csv(PATH_TO_TRAIN_FILE, header=0)

def new_date_features(dataframe: PandasDataFrame) -> None:
    '''Creating new columns on dividing date time stamp

    New features:
    year
    month
    dayofweek
    hour

    :param df: train dataset
    '''
    dataframe['year'] = dataframe.datetime.map(
        lambda x: pd.to_datetime(x).year ).astype(int)
    dataframe['month'] = dataframe.datetime.map(
        lambda x: pd.to_datetime(x).month ).astype(int)
    dataframe['dayofweek'] = dataframe.datetime.map(
        lambda x: pd.to_datetime(x).dayofweek ).astype(int)
    dataframe['hour'] = dataframe.datetime.map(
        lambda x: pd.to_datetime(x).hour ).astype(int)

# new dataframe with new features for dates
new_date_features(train_dataframe)

###############################################################################
# Analyzing dataset using boxplots
###############################################################################
train_boxplot = train_dataframe.drop(
    ['year','casual','registered','count'],
    axis = 1
)

# plot the graph and save it to allfeatures.png
train_boxplot.boxplot()
plt.savefig(os.path.join(PATH_TO_PLOTS, 'all_features.png'))


# count based on weather analysis using boxplot
# we can see max, min, median and outliers from the box plot
train_dataframe.boxplot(column='count', by = 'weather')
plt.savefig(os.path.join(PATH_TO_PLOTS, 'weather_count.png'))

# count based on hour analysis
train_dataframe.boxplot(column='count', by = 'hour')
plt.savefig(os.path.join(PATH_TO_PLOTS, 'hour_count.png'))

# count based dayofweeek analysis
train_dataframe.boxplot(column='count', by = 'dayofweek')
plt.savefig(os.path.join(PATH_TO_PLOTS, 'dayofweek_count.png'))

# mapping day of week to name of week
days = {
    0:'Sunday',
    1:'Monday',
    2:'Tuesday',
    3:'Wednesday',
    4:'Thursday',
    5:'Friday',
    6:'Saturday'
}
train_dataframe['namedayofweek'] = train_dataframe['dayofweek'].map( days )

###############################################################################
# Correlation Matrix
# This matrix tells us if how important the characters are in predicting target
# it also identifies any features that are positively correlated to each other
# So that we can remove correlated features
###############################################################################
print(train_dataframe[
    ['weather','temp','atemp','humidity',
     'windspeed','casual','registered','count']
].corr())

'''Conclusions from the above correlation matrix:

- temp is positively correlated to count
- temp and atemp are higly correlated, should consider only on feature
- when humidity is less there is more count as they are negatively correlated
- when windspeed is less there is more count as they are negatively correlated
'''

###############################################################################
# Preparing tests dataset
###############################################################################
test_dataframe = pd.read_csv(PATH_TO_TEST_FILE, header=0)
new_date_features(test_dataframe)
test_data_timestamp = test_dataframe['datetime']
test_dataframe = test_dataframe.drop(['datetime'], axis = 1)
print("test_dataframe = ", list(test_dataframe))

##############################################################################
# Training Random Forest and Descisoin tree algorithms on training dataset
##############################################################################
# to predict "count"

target_dataframe = train_dataframe['count'] # y
predictors_dataframe_train = train_dataframe.drop(
    ['datetime','casual','registered','count', 'namedayofweek'],
    axis = 1
)

# printing predictors
predictors = list(predictors_dataframe_train)

# Numpy array to give an input to Random forest or Decision Tree
target_dataframe = target_dataframe.values
predictors_dataframe_train = predictors_dataframe_train.values
predictors_dataframe_test = test_dataframe.values

###############################################################################
# Random Forest Classifier
###############################################################################
random_forest = RandomForestClassifier(n_estimators=120)
# fit dataset in random forest
random_forest = random_forest.fit(predictors_dataframe_train, target_dataframe)

###############################################################################
# Decision Tree Classifier
###############################################################################
forest_decisiontree = tree.DecisionTreeClassifier()
forest_decisiontree = forest_decisiontree.fit(
    predictors_dataframe_train,
    target_dataframe
)

###############################################################################
# prediting count and saving it to a file
###############################################################################
count_pred_random_forest = random_forest.predict(
    predictors_dataframe_test
).astype(int)

count_pred_decision_tree = forest_decisiontree.predict(
    predictors_dataframe_test
).astype(int)


def write_to_file(
        filename,
        timestamp_test_data,
        predicted_dataframe
):
    with open(filename, 'w') as fobj:
        fobj_write = csv.writer(fobj)
        fobj_write.writerow(["datetime", "count"])
        fobj_write.writerows(zip(timestamp_test_data, predicted_dataframe))


write_to_file(
    RANDOM_FOREST_FILE,
    test_data_timestamp,
    count_pred_random_forest,
)

write_to_file(
    DECISION_TREE_FILE,
    test_data_timestamp,
    count_pred_decision_tree,
)

###############################################################################
# important features using randome forests
###############################################################################
print(pd.DataFrame(
    random_forest.feature_importances_,
    columns = ["Importance"],
    index = predictors).sort_values(
        ['Importance'],
        ascending = False))


###############################################################################
# RoC curve for randoms forest and decision tree with cross validation
# comparision of models
###############################################################################

# dividing train data set into training and validation data set
train, validation = train_test_split(
    train_dataframe,
    test_size=0.2
)

# preparing training set
training_set_target_value = train["count"]
training_set = train.drop(
    ['datetime', 'casual', 'registered', 'count', 'namedayofweek'],
    axis = 1
)

# preparing validaton set
validation_count = validation["count"]
validation_set = validation.drop(
    ['datetime', 'casual', 'registered', 'count', 'namedayofweek'],
    axis = 1
)

# training your model on training data set
# Numpy array to give an input to Random forest or Decision Tree
training_set_target_value = training_set_target_value.values
training_set = training_set.values
validation_set = validation_set.values

###############################################################################
# Random Forest Classifier
###############################################################################
random_forest = RandomForestClassifier(n_estimators=120)
# fit dataset in random forest
random_forest = random_forest.fit(training_set, training_set_target_value)

###############################################################################
# Decision Tree Classifier
###############################################################################
forest_decisiontree = tree.DecisionTreeClassifier()
forest_decisiontree = forest_decisiontree.fit(
    training_set,
    training_set_target_value
)

###############################################################################
# prediting count and saving it to a file
###############################################################################
validation_pred_random_forest = random_forest.predict(
    validation_set
).astype(int)

validation_pred_decision_tree = forest_decisiontree.predict(
    validation_set
).astype(int)


def write_to_file_validation(
        filename,
        predicted_dataframe
):
    with open(filename, 'w') as fobj:
        fobj_write = csv.writer(fobj)
        fobj_write.writerow(["null","count"])
        fobj_write.writerows(zip('', predicted_dataframe))

write_to_file_validation(
    'validation_RF.csv',
    validation_pred_random_forest,
)

write_to_file_validation(
    'validation_DT.csv',
    validation_pred_decision_tree,
)

write_to_file_validation(
    'validation_actual.csv',
    validation_count
)

def show_confusion_matrix(yt, yp, filename):
    cm = metrics.confusion_matrix(yt, yp)  # Compute confusion matrix
    print('Confusion Matrix for Decision Tree')
    print(cm)
    plt.matshow(cm)  #generate a heatmap of the matrix
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(PATH_TO_PLOTS, filename))
show_confusion_matrix(validation_count, validation_pred_random_forest, 'final_RF_CM.png')
show_confusion_matrix(validation_count, validation_pred_decision_tree, 'final_DT_CM.png')
