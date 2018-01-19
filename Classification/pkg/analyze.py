'''Analyze the dataset to understand the data

This module understands the data on analyzing the dataset

datetime - hourly date + timestamp

season -
1 = spring,
2 = summer,
3 = fall,
4 = winter

holiday - whether the day is considered a holiday

workingday - whether the day is neither a weekend nor holiday

weather -
1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

temp - temperature in Celsius

atemp - "feels like" temperature in Celsius

humidity - relative humidity

windspeed - wind speed

casual - number of non-registered user rentals initiated

registered - number of registered user rentals initiated

count - number of total rentals


how to analyze this dataset

1. how many people rented bike based on season, weather, holiday and working day?
2. what is the count of rental for avg. temp, high temp and low temp?
3. what is the count of rental for avg.windspped, high windspeed and low windspeed?
4. what is the count of rental for avg. humidity , high humidity and low humidity?
5. how many registered users?
6. how many casual users?
7. count based on combinations of the above factors?
8. find the most correlated variables to "count"

Divide the date time to year, month, day, time

1. how many people rented bike in morning, noon and evening?
'''

import os
from typing import List
import csv

import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from pkg.primitives import PATH, PandasDataFrame


def read_training_dataset_pandas(path_to_train_data: PATH):
    return pandas.read_csv(path_to_train_data)

def head_of_dataframe(pandas_dataframe: PandasDataFrame):
    '''Returns first 5 rows'''
    return pandas_dataframe.head()

def tail_of_dataframe(pandas_dataframe: PandasDataFrame):
    '''Returns last 5 rows'''
    return pandas_dataframe.tail()

def nrows_ncolumns(pandas_dataframe: PandasDataFrame):
    '''Returns n rows and n columns using iloc

    This function returns first 5 rows where rows are labled from
    0,1,23...
    '''
    return pandas_dataframe.iloc[0:5, :]

def labled_columns(pandas_dataframe: PandasDataFrame):
    '''Returns a pandas dataframe for required cols'''
    return pandas_dataframe.loc[0:5, ["season", "count"]]

def generate_day(pandas_dateframe: PandasDataFrame,
                 date_field: str,
):
    '''Return a new data frame with month, day, year, time'''
    pandas_dataframe['year'] = pandas_dataframe[date_field].dt.year
    pandas_dataframe['month'] = pandas_dataframe[date_field].dt.month
    pandas_dataframe['day'] = pandas_dataframe[date_field].dt.day
    pandas_dataframe['time'] = pandas_dataframe[date_field].dt.time
    return pandas_dataframe

def generate_day(pandas_dateframe: PandasDataFrame,
                    date_field: str,
):
    '''Return a new data frame with month, day, year, time'''
    pandas_dataframe[new_column] = pandas_dataframe[date_field].dt.day



def groupby_sum_dataframe(
        pandas_dataframe: PandasDataFrame,
        independent_variable: str,
        class_label: str,
) -> PandasDataFrame:
    '''Returns PandasDataFrame

    with sum of class_label grouped by dependent column

    :param pandas_dataframe: train dataset as pandas dataframe
    :param independent_variable: column by which dataset is grouped by
    :param class_label: class label for dataset
    '''
    return pandas_dataframe.groupby(independent_variable)[class_label].sum()


def plotting(
        pandas_dataframe: PandasDataFrame,
        independent_variable: str,
        class_label: str,
        save_plot_loc: PATH,
        filename: str
):
    '''Plotting bar graph for pandas dataframe

    :param pandas_dataframe: pandas dataframe
    :param independent_variable: column by which dataset is groupbed by
    :param class_label: class label for dataset
    :param save_plot_loc: location to which plot will be saved
    :param filename: filename of the .png file
    '''
    df = groupby_sum_dataframe(
        pandas_dataframe,
        independent_variable,
        class_label,
    )
    df.plot.bar()
    plt.savefig(os.path.join(save_plot_loc, filename))

def correlation(
        pandas_dataframe: PandasDataFrame,
):
    '''Return correlation matrix

    This matrix contains correlation between variables, this is useful
    to filter out most positively correlated variables


    temp and atemp are correlated to each other so only one varaible can
    be considered in the model
    '''
    return pandas_dataframe.corr()
