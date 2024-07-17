'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name:     zID:

Name:     zID:
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

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global state
    #global params

    # TODO: Add code to generate your chosen actions, using the current state and sensor_data

    actions_dict = {'lights1': 'on', 'lights2': 'on', 'lights3': 'on', 'lights4': 'on', 
                    'lights5': 'on', 'lights6': 'on', 'lights7': 'on', 'lights8': 'on', 
                    'lights9': 'on', 'lights10': 'on', 'lights11': 'on', 'lights12': 'on',
                    'lights13': 'on', 'lights14': 'on', 'lights15': 'on', 'lights16': 'on', 
                    'lights17': 'on', 'lights18': 'on', 'lights19': 'on', 'lights20': 'on',
                    'lights21': 'on', 'lights22': 'on', 'lights23': 'on', 'lights24': 'on',
                    'lights25': 'on', 'lights26': 'on', 'lights27': 'on', 'lights28': 'on',
                    'lights29': 'on', 'lights30': 'on', 'lights31': 'on', 'lights32': 'on',
                    'lights33': 'on', 'lights34': 'on'}
    return actions_dict
