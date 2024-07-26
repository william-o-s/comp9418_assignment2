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

# column values: motion / no motion
REGEX_MOTION_SENSOR = re.compile(r'^motion_sensor[0-9]*$')
# column values: tuple(room, # of people)
REGEX_ROBOT = re.compile(r'^robot[0-9]*$')
# column values: date time
REGEX_TIME = re.compile(r'^time$')
# column values: # of people
REGEX_CAMERA = r'^camera[0-9]*$'
REGEX_DOOR_SENSOR = r'^door_sensor[0-9]*$'
REGEX_ROOM = r'^r[0-9]*$'
REGEX_CORRIDOR = r'^c[0-9]*$'
REGEX_OUTSIDE = r'^outside$'
REGEX_PEOPLE_COUNT = re.compile('|'.join([REGEX_CAMERA, REGEX_DOOR_SENSOR, REGEX_ROOM, REGEX_CORRIDOR, REGEX_OUTSIDE]))

def parse_str_to_time(time_str: str) -> dt.datetime:
    return dt.datetime.strptime(time_str, '%H:%M:%S')

def setup_training_data(filename: Literal['data1.csv', 'data2.csv']) -> pd.DataFrame:
    df = pd.read_csv(filename, header=[0], index_col=[0])

    # Set all columns
    for col in df.columns:
        if REGEX_MOTION_SENSOR.match(col):
            pass
        elif REGEX_ROBOT.match(col):
            # Replace with actual tuple and convert to boolean
            # df[col] = df[col].apply(ast.literal_eval)
            # df[col] = df[col].apply(lambda x: (x[0], 'on' if x[1] > 0 else 'off'))
            pass
        elif REGEX_TIME.match(col):
            df[col] = df[col].apply(parse_str_to_time)
        elif REGEX_PEOPLE_COUNT.match(col):
            # Replace with buckets
            df[col] = df[col].astype(object)
            df[col] = df[col].apply(Utils.bucket_people_count)
    return df

def process_sensor_data(sensor_data: dict[str]) -> dict[str]:
    new_sensor_data = copy.deepcopy(sensor_data)

    # Set all data
    for var_name, data in new_sensor_data.items():
        if not data:
            continue    # IGNORE NONETYPE LIKE GUSTAVO SAYS

        if REGEX_TIME.match(var_name) and isinstance(data, str):
            new_sensor_data[var_name] = parse_str_to_time(data)
        elif REGEX_PEOPLE_COUNT.match(var_name):
            new_sensor_data[var_name] = Utils.bucket_people_count(data)

    return new_sensor_data

###################################
# Setup

training_data = setup_training_data('data1.csv')

remap_count_outcome = {
    var_name: Utils.PEOPLE_COUNT_BUCKETS
    for var_name in training_data.columns
    if REGEX_PEOPLE_COUNT.match(var_name)
}
outcomes_remap: dict[str, tuple] = { **remap_count_outcome, 'time': Utils.TIME_BUCKETS }

room1 = RoomPredictor(training_data, 'r1', ['motion_sensor1'], outcomes_remap)
room14 = RoomPredictor(training_data, 'r14', ['motion_sensor2'], outcomes_remap)
room19 = RoomPredictor(training_data, 'r19', ['motion_sensor3'], outcomes_remap)
room28 = RoomPredictor(training_data, 'r28', ['motion_sensor4'], outcomes_remap)
room29 = RoomPredictor(training_data, 'r29', ['motion_sensor5'], outcomes_remap)
room32 = RoomPredictor(training_data, 'r32', ['motion_sensor6'], outcomes_remap)

room3 = RoomPredictor(training_data, 'r3', ['camera1'], outcomes_remap)
room21 = RoomPredictor(training_data, 'r21', ['camera2'], outcomes_remap)
room25 = RoomPredictor(training_data, 'r25', ['camera3'], outcomes_remap)
room34 = RoomPredictor(training_data, 'r34', ['camera4'], outcomes_remap)

###################################
# CONFIG

threshold = 0.99
print('Threshold: ' + str(threshold))

def get_action(sensor_data: dict[str]):
    # declare state as a global variable so it can be read and modified within this function
    global state
    # global params

    sensor_data = process_sensor_data(sensor_data)

    # TODO: Add code to generate your chosen actions, using the current state and sensor_data

    room1_prediction = room1.prediction(threshold=threshold, motion=sensor_data['motion_sensor1'])
    room14_prediction = room14.prediction(threshold=threshold, motion=sensor_data['motion_sensor2'])
    room19_prediction = room19.prediction(threshold=threshold, motion=sensor_data['motion_sensor3'])
    room28_prediction = room28.prediction(threshold=threshold, motion=sensor_data['motion_sensor4'])
    room29_prediction = room29.prediction(threshold=threshold, motion=sensor_data['motion_sensor5'])
    room32_prediction = room32.prediction(threshold=threshold, motion=sensor_data['motion_sensor6'])
    room3_prediction = room3.prediction(threshold=threshold, camera=sensor_data['camera1'])
    room21_prediction = room21.prediction(threshold=threshold, camera=sensor_data['camera2'])
    room25_prediction = room25.prediction(threshold=threshold, camera=sensor_data['camera3'])
    room34_prediction = room34.prediction(threshold=threshold, camera=sensor_data['camera4'])

    actions_dict = {
        'lights1': room1_prediction, 
        'lights2': 'on', 
        'lights3': room3_prediction, 
        'lights4': 'on', 
        'lights5': 'on', 
        'lights6': 'on', 
        'lights7': 'on', 
        'lights8': 'on', 
        'lights9': 'on', 
        'lights10': 'on', 
        'lights11': 'on', 
        'lights12': 'on',
        'lights13': 'on', 
        'lights14': room14_prediction, 
        'lights15': 'on', 
        'lights16': 'on', 
        'lights17': 'on', 
        'lights18': 'on', 
        'lights19': room19_prediction, 
        'lights20': 'on',
        'lights21': room21_prediction, 
        'lights22': 'on', 
        'lights23': 'on', 
        'lights24': 'on',
        'lights25': room25_prediction, 
        'lights26': 'on', 
        'lights27': 'on', 
        'lights28': room28_prediction,
        'lights29': room29_prediction, 
        'lights30': 'on', 
        'lights31': 'on', 
        'lights32': room32_prediction,
        'lights33': 'on', 
        'lights34': room34_prediction
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
    
    robot_action(sensor_data['robot1'])
    robot_action(sensor_data['robot2'])

    return actions_dict
