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
import datetime as dt
import sklearn
import ast
import re
import pickle
import json

# Required libraries
import re
from typing import Literal
from MF_RoomPredictor import RoomPredictor
import MF_Utils as Utils

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
        if Utils.REGEX_MOTION_SENSOR.match(col):
            pass
        elif Utils.REGEX_ROBOT.match(col):
            # Replace with actual tuple and convert to boolean
            # df[col] = df[col].apply(ast.literal_eval)
            # df[col] = df[col].apply(lambda x: (x[0], 'on' if x[1] > 0 else 'off'))
            pass
        elif Utils.REGEX_TIME.match(col):
            df[col] = df[col].apply(Utils.bucket_time_of_day)
        elif Utils.REGEX_PEOPLE_COUNT.match(col):
            # Replace with buckets
            df[col] = df[col].astype(object)
            df[col] = df[col].apply(Utils.bucket_people_count)
    return df

def process_sensor_data(sensor_data: dict[str]) -> dict[str]:
    new_sensor_data = {}

    # Set all data
    for var_name, data in sensor_data.items():
        if data is None:
            continue    # IGNORE NONETYPE LIKE GUSTAVO SAYS

        if Utils.REGEX_TIME.match(var_name):
            new_sensor_data[var_name] = Utils.bucket_time_of_day(data)
        elif Utils.REGEX_PEOPLE_COUNT.match(var_name):
            new_sensor_data[var_name] = Utils.bucket_people_count(data)
        else:
            new_sensor_data[var_name] = data

    return new_sensor_data

###################################
# Setup

training_data = setup_training_data('data1.csv')

# Define rooms and their evidence
room_evidences = {
    'r1'    : ['motion_sensor1'],
    'r14'   : ['motion_sensor2'],
    'r19'   : ['motion_sensor3'],
    # 'r28'   : ['motion_sensor4', 'door_sensor4'],
    'r28'   : ['motion_sensor4'],
    'r29'   : ['motion_sensor5'],
    'r32'   : ['motion_sensor6'],
    # 'r3'    : ['camera1', 'door_sensor1'],
    'r3'    : ['camera1'],
    'r21'   : ['camera2'],
    'r25'   : ['camera3'],
    'r34'   : ['camera4'],
    # 'r2': ['door_sensor1'],
    # 'r20': ['door_sensor3'],
    # 'r26': ['door_sensor3'],
}

# Manual repeated entries
for evidence in room_evidences.values():
    evidence.append('time')

# Remap outcomes for each room
remap_count_outcome = {
    var_name: Utils.PEOPLE_COUNT_BUCKETS
    for var_name in training_data.columns
    if Utils.REGEX_PEOPLE_COUNT.match(var_name)
}
outcomes_remap: dict[str, tuple] = { **remap_count_outcome, 'time': Utils.TIME_BUCKETS }

room_predictors = {
    str('lights' + str(room[1:])): RoomPredictor(training_data, room, evidence, outcomes_remap)
    for room, evidence in room_evidences.items()
}

###################################
# CONFIG

threshold = 0.5
print('Threshold: ' + str(threshold))

def get_action(sensor_data: dict[str]):
    # declare state as a global variable so it can be read and modified within this function
    global state
    # global params

    sensor_data = process_sensor_data(sensor_data)

    # TODO: Add code to generate your chosen actions, using the current state and sensor_data

    lights_labels = ['lights' + str(i) for i in range(1, 35)]
    actions_dict = {
        light: room_predictors[light].prediction(threshold=threshold, **sensor_data)    # NOTE: this has a side effect! Do not call prediction multiple times in the same iteration!
        if light in room_predictors else 'on'   # if no predictor, just force on
        for light in lights_labels
    }

    # Convert robot predictions from str to tuple
    def extract_tuple(robot):
        room = re.search("(?<=')\w+", robot).group()
        count = re.search("(?<=\s)\d+", robot).group()
        return room, count
    
    def robot_action(robot) -> None:
        if robot is None or robot == 'None':
            return
        
        room, people = extract_tuple(robot)
        if room is not None and room.startswith('r'):
            light = room.replace('r', 'lights')
            actions_dict[light] = 'on' if int(people) > 0 else 'off'
    
    if 'robot1' in sensor_data:
        robot_action(sensor_data['robot1'])
    if 'robot2' in sensor_data:
        robot_action(sensor_data['robot2'])

    return actions_dict

# print(get_action({ 'motion_sensor1': 'no motion' }))
