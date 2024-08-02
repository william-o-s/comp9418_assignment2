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

    # set up dataframe to use the past value of neighbouring rooms
    shifted = df.shift(1)
    # fill in t-1 bracket as no one there
    shifted.iloc[0,17:53] = '0'
    shifted.iloc[0,53] = '>=10'

    # rename these columns
    names = shifted.iloc[:,17:54].columns
    names = [i+'_last' for i in names]
    shifted = shifted.iloc[:,17:54]
    shifted.columns = names

    #combine data
    combo = pd.concat([df, shifted], axis =1)
    return combo

def process_sensor_data(sensor_data: dict[str]) -> dict[str]:
    new_sensor_data = {}

    # Set all data
    for var_name, data in sensor_data.items():
        if data is None:
            continue    # IGNORE NONETYPE LIKE GUSTAVO SAYS

        if Utils.REGEX_TIME.match(var_name):
            new_sensor_data[var_name] = Utils.bucket_time_of_day(data)
        elif Utils.REGEX_PEOPLE_COUNT.match(var_name):
            new_sensor_data[var_name] = Utils.bucket_people_count(int(data))
        else:
            new_sensor_data[var_name] = data

    return new_sensor_data

###################################
# Setup

data1_training_data = setup_training_data('data1.csv')
data2_training_data = setup_training_data('data2.csv')

# training_data = pd.concat([data1_training_data, data2_training_data], axis=0)
#training_data = data1_training_data
training_data = data2_training_data
#print(training_data.shape)

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
    'r2'    : [],
    'r4'    : [],
    'r5'    : [],
    'r6'    : [],
    'r7'    : [],
    'r8'    : [],
    'r9'    : [],
    'r10'   : [],
    'r11'   : [],
    'r12'   : [],
    'r13'   : [],
    'r15'   : [],
    'r16'   : [],
    'r17'   : [],
    'r18'   : [],
    'r20'   : [],
    'r22'   : [],
    'r23'   : [],
    'r24'   : [],
    'r26'   : [],
    'r27'   : [],
    'r30'   : [],
    'r31'   : [],
    'r33'   : [],
    'c1'   : [],   
  #  'c2'   : [],
}

# comments bc i was going to remove corridors for a test
# pretending c2 doesn't exist so that my computer doesnt die
room_adj_ls = {'r1': ['r2_last'],
 'r3': ['r2_last', 'r12_last'],
 'c2': ['r12_last',
  'r11_last',
  'r17_last',
  'r18_last',
  'r13_last',
  'r9_last',
  'r10_last',
  'r16_last',
  'r15_last',
  'r7_last',
  'r8_last',
  'c1_last'],
 'r14': ['c1_last', 'r6_last', 'r22_last'],
 'r14': ['r6_last', 'r22_last'],

 'r5': ['r6_last'],
 'r4': ['r6_last'],
 'r24': ['r22_last', 'r28_last'],
 'r34': ['r33_last', 'r28_last'],
 'c1': ['r28_last', 'r27_last', 'r21_last', 'c2_last', 'r14_last'],
 #'c1': ['r28_last', 'r27_last', 'r21_last', 'r14_last'],

 'r26': ['r27_last', 'r20_last', 'r30_last', 'r25_last'],
 'r19': ['r23_last', 'r20_last'],
 'r23': ['r20_last', 'r19_last'],
 'r21': ['r27_last', 'c1_last'],
#   'r21': ['r27_last'],

 'r29': ['r30_last', 'r25_last'],
 'r32': ['r27_last', 'r31_last'],
 'r2': ['r1_last', 'r3_last'],
 'r12': ['r3_last', 'c2_last'],
# 'r12': ['r3_last'],
 'r11': ['c2_last'],
 'r17': ['c2_last'],
 'r18': ['c2_last'],
 'r13': ['c2_last'],
 'r9': ['c2_last'],
 'r10': ['c2_last'],
 'r16': ['c2_last'],
 'r15': ['c2_last'],
 'r7': ['c2_last'],
 'r8': ['c2_last'],
#  'r11': [],
#  'r17': [],
#  'r18': [],
#  'r13': [],
#  'r9': [],
#  'r10': [],
#  'r16': [],
#  'r15': [],
#  'r7': [],
#  'r8': [],
 'r6': ['r14_last', 'r5_last', 'r4_last'],
 'r22': ['r14_last', 'r24_last'],
 'r28': ['r24_last', 'r34_last', 'c1_last'],
 'r33': ['r34_last'],
 'r27': ['c1_last', 'r26_last', 'r21_last', 'r32_last'],
 'r20': ['r26_last', 'r19_last', 'r23_last'],
 'r30': ['r26_last', 'r29_last'],
 'r25': ['r26_last', 'r29_last'],
 'r31': ['r32_last']}

# Manual repeated entries
for key, evidence in room_evidences.items():
    # evidence.append('time')
    evidence.extend(room_adj_ls[key])

# initialise state var
state = {} 
for key in room_evidences:
    state[key]='0'


# Remap outcomes for each room
remap_count_outcome = {
    var_name: Utils.PEOPLE_COUNT_BUCKETS
    for var_name in training_data.columns
    if Utils.REGEX_PEOPLE_COUNT.match(var_name)
}

outcomes_remap: dict[str, tuple] = { **remap_count_outcome, 'time': Utils.TIME_BUCKETS }

# room_predictors = {
#     str('lights' + str(room[1:])): RoomPredictor(training_data, room, evidence, outcomes_remap)
#     for room, evidence in room_evidences.items()
# }
# print('outcomes remap', outcomes_remap)
# print('room evidences', room_evidences)

# room_predictors = {
#     room: RoomPredictor(training_data, room, evidence, outcomes_remap)
#     for room, evidence in room_evidences.items()
# }
room_predictors = {}
for room, evidence in room_evidences.items():

    room_predictors[room]= RoomPredictor(training_data, room, evidence, outcomes_remap)
###################################
# CONFIG

threshold = .9 #0.5
print('Threshold: ' + str(threshold))

room_labels = [key for key in room_evidences]


def get_action(sensor_data: dict[str]):
    # declare state as a global variable so it can be read and modified within this function
    global state
    # global params

    sensor_data = process_sensor_data(sensor_data)
    for key, val in state.items():
        newkey = key + '_last'
        sensor_data[newkey]=val


    # TODO: Add code to generate your chosen actions, using the current state and sensor_data

    lights_labels = ['lights' + str(i) for i in range(1, 35)]

    # this now returns a tuple - (prediction_output, lights)
    all = {
        room: room_predictors[room].prediction(threshold=threshold, **sensor_data)
        for room in room_labels
    }
    # new state variables
    state = {room: all[room][0] for room in room_labels}

    # actions_dict = {
    #     light: room_predictors[light].prediction(threshold=threshold, **sensor_data)    # NOTE: this has a side effect! Do not call prediction multiple times in the same iteration!
    #     if light in room_predictors else 'on'   # if no predictor, just force on
    #     for light in lights_labels
    # }
    actions_dict = {}
    for room in room_labels:
        if room not in ['c1', 'c2']:
            light_label = str('lights' + str(room[1:]))
            actions_dict[light_label]= all[room][1]

    # Convert robot predictions from str to tuple
    def extract_tuple(robot):
        room = re.search("(?<=')\w+", robot).group()
        count = re.search("(?<=\s)\d+", robot).group()
        return room, count
    
    def robot_action(robot) -> None:
        if robot is None or robot == 'None':
            return
        
        room, people = extract_tuple(robot)
        if room is not None:
            # record for next get action 
            state[room]= Utils.bucket_people_count(int(people))

            if room.startswith('r'):
                light = room.replace('r', 'lights')
                actions_dict[light] = 'on' if int(people) > 0 else 'off'
                for outcome in room_predictors[room].hmm.state.outcome_space[room]:
                    if outcome == state[room]:
                        room_predictors[room].hmm.state[outcome]=1
                    else:
                        room_predictors[room].hmm.state[outcome]=0
    
    if 'robot1' in sensor_data:
        robot_action(sensor_data['robot1'])
    if 'robot2' in sensor_data:
        robot_action(sensor_data['robot2'])

    return actions_dict

# print(get_action({ 'motion_sensor1': 'no motion' }))
