'''Reinforcement Learning'''

import os
import math
import random
import pdb

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# reading data as pandas dataframe from csv file

creatives = pd.read_csv('ucb_data.csv')
print(creatives.head(10))

# Initialization
user = creatives.user
creatives = creatives.drop(['user'], axis = 1)
N = 10000
d = 6
total_reward_rand = 0

ad_selected = []
for i in range(0,N):
    ad = random.randrange(d)
    ad_selected.append(ad)
    reward = creatives.values[i,ad]
    total_reward_rand = total_reward_rand + reward

print(f'total clicks without any strategy: {total_reward_rand}')
plt.hist(ad_selected)
plt.savefig('no_ucb.png')
# applying UCB algorithms on this dataset to post ads based on strategy

N = 10000
d = 6
ad_selected = []
no_selection = [0] * d
sum_reward = [0] * d
total_reward_ucb = 0

for i in range(0,N):
    max_upper_bnd = 0
    ad = 0
    for j in range(0,d):
        if (no_selection[j] > 0):
            avg_reward = sum_reward[j]/no_selection[j]
            delta_j = (math.sqrt(3/2) * math.log(i+1)/no_selection[j])
            upper_bnd = avg_reward + delta_j
        else:
            upper_bnd = 120000
        if upper_bnd > max_upper_bnd:
            max_upper_bnd = upper_bnd
            ad = j
    ad_selected.append(ad)
    no_selection[ad] = no_selection[ad] + 1
    reward = creatives.values[i,ad]
    sum_reward[ad]= sum_reward[ad] + reward
    total_reward_ucb = total_reward_ucb + reward

print(f'total clicks with UCB algorithm {total_reward_ucb}')
plt.hist(ad_selected)
plt.savefig('ucb.png')
