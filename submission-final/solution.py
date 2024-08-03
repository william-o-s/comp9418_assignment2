'''
COMP9418 Assignment 2
Team: Martin and Gustavo's Fans

Name: Freya Stevens     zID: z5446846
Name: William Setiawan  zID: z5388080
'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

from typing import Literal

# Allowed libraries
import re

import pandas as pd

# Required libraries
from MF_RoomPredictor import RoomPredictor
import MF_Utils as Utils

###################################
#
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
#

###################################
# Training data

def setup_training_data(filename: Literal['data1.csv', 'data2.csv']) -> pd.DataFrame:
    '''Formats training data into required data.'''
    df = pd.read_csv(filename, header=[0], index_col=[0])

    # Set all columns
    for col in df.columns:
        if Utils.REGEX_MOTION_SENSOR.match(col):
            # Data already in binary format
            pass
        elif Utils.REGEX_ROBOT.match(col):
            # Tuple parsed later
            pass
        elif Utils.REGEX_TIME.match(col):
            # Technically not necessary
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
    '''Process sensor data to match training data'''
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
# Setup training data

data2_training_data = setup_training_data('data2.csv')
training_data = data2_training_data

# Stub rooms
room_labels = ['r' + str(i) for i in range(1, 35)]
room_labels.append('c1')
room_evidences = { room_label: [] for room_label in room_labels }

# Update rooms with specific evidence using |= update operator
room_evidences |= {
    # NOTE: redefining keys below replaces evidence
    'r1'    : ['motion_sensor1'],
    'r14'   : ['motion_sensor2'],
    'r19'   : ['motion_sensor3'],
    'r28'   : ['motion_sensor4'],
    'r29'   : ['motion_sensor5'],
    'r32'   : ['motion_sensor6'],
    'r3'    : ['camera1'],
    'r21'   : ['camera2'],
    'r25'   : ['camera3'],
    'r34'   : ['camera4'],
}

# comments bc i was going to remove corridors for a test
# pretending c2 doesn't exist so that my computer doesnt die
room_adj_ls = {
    'r1': ['r2_last'],
    'r2': ['r1_last', 'r3_last'],
    'r3': ['r2_last', 'r12_last'],
    'r4': ['r6_last'],
    'r5': ['r6_last'],
    'r6': ['r14_last', 'r5_last', 'r4_last'],
    'r7': ['c2_last'],
    'r8': ['c2_last'],
    'r9': ['c2_last'],
    'r10': ['c2_last'],
    'r11': ['c2_last'],
    'r12': ['r3_last', 'c2_last'],
    'r13': ['c2_last'],
    'r14': ['r6_last', 'r22_last'],
    'r15': ['c2_last'],
    'r16': ['c2_last'],
    'r17': ['c2_last'],
    'r18': ['c2_last'],
    'r19': ['r23_last', 'r20_last'],
    'r20': ['r26_last', 'r19_last', 'r23_last'],
    'r21': ['r27_last', 'c1_last'],
    'r22': ['r14_last', 'r24_last'],
    'r23': ['r20_last', 'r19_last'],
    'r24': ['r22_last', 'r28_last'],
    'r25': ['r26_last', 'r29_last'],
    'r26': ['r27_last', 'r20_last', 'r30_last', 'r25_last'],
    'r27': ['c1_last', 'r26_last', 'r21_last', 'r32_last'],
    'r28': ['r24_last', 'r34_last', 'c1_last'],
    'r29': ['r30_last', 'r25_last'],
    'r30': ['r26_last', 'r29_last'],
    'r31': ['r32_last'],
    'r32': ['r27_last', 'r31_last'],
    'r33': ['r34_last'],
    'r34': ['r33_last', 'r28_last'],
    'c1': ['r28_last', 'r27_last', 'r21_last', 'c2_last', 'r14_last'],
    'c2': [
        'r12_last',
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
        'c1_last'
    ],
}

# Add further evidence from adjacent rooms
for key, evidence in room_evidences.items():
    evidence.extend(room_adj_ls[key])

# initialise state var
state = {}
for key in room_evidences:
    state[key] = '0'

# Remap outcomes for each room
remap_count_outcome = {
    var_name: Utils.PEOPLE_COUNT_BUCKETS
    for var_name in training_data.columns
    if Utils.REGEX_PEOPLE_COUNT.match(var_name)
}

outcomes_remap: dict[str, tuple] = { **remap_count_outcome, 'time': Utils.TIME_BUCKETS }

room_predictors = {}
for room, evidence in room_evidences.items():
    room_predictors[room]= RoomPredictor(training_data, room, evidence, outcomes_remap)

###################################
# CONFIG

THRESHOLD = .9
room_labels = list(room_evidences)

def get_action(sensor_data: dict[str]):
    '''Generate your chosen actions, using the current state and sensor_data'''

    # declare state as a global variable so it can be read and modified within this function
    global state

    sensor_data = process_sensor_data(sensor_data)
    for room_label, room_state in state.items():
        newkey = room_label + '_last'
        sensor_data[newkey] = room_state

    # this now returns a tuple - (prediction_output, lights)
    # NOTE: this has a side effect! Do not call prediction multiple times in the same iteration!
    all_room_preds = {
        room: room_predictors[room].prediction(threshold=THRESHOLD, **sensor_data)
        for room in room_labels
    }
    # new state variables
    state = {room: all_room_preds[room][0] for room in room_labels}

    actions_dict = {}
    for room_label in room_labels:
        if room_label not in ['c1', 'c2']:
            light_label = str('lights' + str(room_label[1:]))
            actions_dict[light_label]= all_room_preds[room_label][1]

    # Convert robot predictions from str to tuple
    def extract_tuple(robot):
        robot_room = re.search(r"(?<=')\w+", robot).group()
        robot_count = re.search(r"(?<=\s)\d+", robot).group()
        return robot_room, robot_count

    def robot_action(robot) -> None:
        if robot is None or robot == 'None':
            return

        robot_room, robot_people = extract_tuple(robot)
        if robot_room is not None:
            # record for next get action
            state[robot_room]= Utils.bucket_people_count(int(robot_people))

            if robot_room.startswith('r'):
                light = robot_room.replace('r', 'lights')
                actions_dict[light] = 'on' if int(robot_people) > 0 else 'off'
                for outcome in room_predictors[robot_room].hmm.state.outcome_space[robot_room]:
                    if outcome == state[robot_room]:
                        room_predictors[robot_room].hmm.state[outcome]=1
                    else:
                        room_predictors[robot_room].hmm.state[outcome]=0

    if 'robot1' in sensor_data:
        robot_action(sensor_data['robot1'])
    if 'robot2' in sensor_data:
        robot_action(sensor_data['robot2'])

    return actions_dict
