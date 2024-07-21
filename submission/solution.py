'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: Freya Stevens     zID: z5446846

Name: William Setiawan  zID: z5388080
'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries 
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re
import pickle
import json

# Required libraries
from typing import Literal
from RoomPredictor import RoomPredictor

###################################
# Code stub
# 
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
# 


# this global state variable demonstrates how to keep track of information over multiple 
# calls to get_action 
state = {} 

# params = pd.read_csv(...)

###################################
# Training data

def setup_training_data(filename: Literal['data1.csv', 'data2.csv']) -> pd.DataFrame:
    df = pd.read_csv(filename, header=[0], index_col=[0])

    # Set all columns
    for col in df.columns:
        if col.startswith('motion_sensor'):
            # Replace
            # df.loc[df[col] == 'motion', col] = 'on'
            # df.loc[df[col] == 'no motion', col] = 'off'
            pass
        elif col.startswith('robot'):
            # Replace second value with boolean
            # df[col] = df[col].apply(lambda x: (x[0], 'on' if x[1] > 0 else 'off'))
            pass
        elif (col.startswith('camera')
                or col.startswith('door_sensor')
                or col.startswith('r')
                or col.startswith('c')):
            # Replace with boolean
            df.loc[df[col] > 0, col] = 'on'
            df.loc[df[col] == 0, col] = 'off'

    return df


###################################
# Setup

training_data = setup_training_data('data1.csv')
room1 = RoomPredictor(training_data, 'r1', ['motion_sensor1'])
room14 = RoomPredictor(training_data, 'r14', ['motion_sensor2'])
room19 = RoomPredictor(training_data, 'r19', ['motion_sensor3'])
room28 = RoomPredictor(training_data, 'r28', ['motion_sensor4'])
room29 = RoomPredictor(training_data, 'r29', ['motion_sensor5'])
room32 = RoomPredictor(training_data, 'r32', ['motion_sensor6'])

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global state
    #global params

    # TODO: Add code to generate your chosen actions, using the current state and sensor_data
    room1_prediction = room1.prediction(motion=sensor_data['motion_sensor1'])
    room14_prediction = room14.prediction(motion=sensor_data['motion_sensor2'])
    room19_prediction = room19.prediction(motion=sensor_data['motion_sensor3'])
    room28_prediction = room28.prediction(motion=sensor_data['motion_sensor4'])
    room29_prediction = room29.prediction(motion=sensor_data['motion_sensor5'])
    room32_prediction = room32.prediction(motion=sensor_data['motion_sensor6'])

    actions_dict = {'lights1': room1_prediction, 'lights2': 'on', 'lights3': 'on', 'lights4': 'on', 
                    'lights5': 'on', 'lights6': 'on', 'lights7': 'on', 'lights8': 'on', 
                    'lights9': 'on', 'lights10': 'on', 'lights11': 'on', 'lights12': 'on',
                    'lights13': 'on', 'lights14': room14_prediction, 'lights15': 'on', 'lights16': 'on', 
                    'lights17': 'on', 'lights18': 'on', 'lights19': room19_prediction, 'lights20': 'on',
                    'lights21': 'on', 'lights22': 'on', 'lights23': 'on', 'lights24': 'on',
                    'lights25': 'on', 'lights26': 'on', 'lights27': 'on', 'lights28': room28_prediction,
                    'lights29': room29_prediction, 'lights30': 'on', 'lights31': 'on', 'lights32': room32_prediction,
                    'lights33': 'on', 'lights34': 'on'}
    return actions_dict
