import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import product
from BayesNet_VE import BayesNet
from DiscreteFactors import Factor
from Graph import Graph
from HiddenMarkovModel import HiddenMarkovModel

def all_equal_this_index(dict_of_arrays, **fixed_vars):
    # base index is a boolean vector, everywhere true
    first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
    index = np.ones_like(first_array, dtype=np.bool_)
    for var_name, var_val in fixed_vars.items():
        index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
    return index

def estimate_factor(data, var_name, parent_names, outcome_space, alpha=2):
    var_outcomes = outcome_space[var_name]
    parent_outcomes = [outcome_space[var] for var in (parent_names)]
    # cartesian product to generate a table of all possible outcomes
    all_parent_combinations = product(*parent_outcomes)

    f = Factor(list(parent_names)+[var_name], outcome_space)
    
    for i, parent_combination in enumerate(all_parent_combinations):
        parent_vars = dict(zip(parent_names, parent_combination))
        parent_index = all_equal_this_index(data, **parent_vars)
        for var_outcome in var_outcomes:
            var_index = (np.asarray(data[var_name])==var_outcome)
            f[tuple(list(parent_combination)+[var_outcome])] = ((var_index & parent_index).sum() + alpha) / (parent_index.sum() + alpha - 1)
            
    return f

class RoomPredictor:
    def __init__(self, data: pd.DataFrame, room: str, sensors: list[str]) -> None:
        # Avoid door_sensor and robot sensors for now
        assert all(not (x.startswith('door_sensor') or x.startswith('robot')) for x in sensors)

        # Setup training data
        self.room = room
        self.sensors = sensors
        self.vars = [room] + sensors
        self.training_data = data[self.vars]
        
        # Setup predictor
        self.outcome_space = self.learn_outcome_space()
        self.state_factor = self.learn_states()
        self.transition_factor = self.learn_transitions()
        self.emission_factor = self.learn_emissions()
        self.var_remap = { str(self.room + '_next'): self.room }

        self.hmm = HiddenMarkovModel(self.state_factor, self.transition_factor, self.emission_factor, self.var_remap)

    def prediction(self, threshold=0.95, **evidence):
        prediction_factor: Factor = self.hmm.forward(normalize=True, **evidence)

        if threshold is not None:
            if prediction_factor['off'] >= threshold:
                return 'off'
            return 'on'

        mle_index = prediction_factor.table.argmax()
        prediction = self.outcome_space[self.room][mle_index]
        return prediction

    def learn_outcome_space(self) -> dict:
        outcome_space = {}
        for column in list(self.training_data.columns):
            values = list(self.training_data[column].unique())
            outcome_space[column] = tuple(values)
        
        outcome_space[self.room + '_next'] = outcome_space[self.room]

        return outcome_space

    def learn_states(self) -> Factor:
        state_factor = Factor((self.room,), self.outcome_space)
        probs = self.training_data[self.room].value_counts(normalize=True)
        
        for outcome in self.outcome_space[self.room]:
            state_factor[outcome] = probs[outcome]
        
        return state_factor

    def learn_transitions(self) -> Factor:
        transition_factor = Factor((self.room, self.room + '_next'), self.outcome_space)

        room_data = self.training_data[self.room]

        # Get total counts
        transition_counts = defaultdict(int)
        len_data = len(room_data)

        for i in range(len_data - 1):
            transition_counts[room_data[i], room_data[i+1]] += 1

        # Normalize probabilities
        for key in transition_counts:
            transition_counts[key] = transition_counts[key] / len_data
        
        transition_factor['on', 'on'] = transition_counts[('on', 'on')]
        transition_factor['on', 'off'] = transition_counts[('on', 'off')]
        transition_factor['off', 'on'] = transition_counts[('off', 'on')]
        transition_factor['off', 'off'] = transition_counts[('off', 'off')]
        
        return transition_factor

    def learn_emissions(self, alpha=2) -> Factor:
        return estimate_factor(self.training_data, self.room, self.sensors, self.outcome_space, alpha)
    
    