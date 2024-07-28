from collections import defaultdict
from itertools import permutations

import pandas as pd
import numpy as np
import MF_Utils as Utils
from MF_BayesNet_VE import BayesNet
from MF_DiscreteFactors import Factor
from MF_Graph import Graph
from MF_HiddenMarkovModel import HiddenMarkovModel

class RoomPredictor:
    def __init__(
            self,
            data: pd.DataFrame,
            room: str,
            sensors: list[str],
            room_adj_ls: dict,
            outcomes_remap: dict[str, tuple] = None)-> None:
        # Avoid door_sensor for now
        assert all(not x.startswith('door_sensor') for x in sensors)

        # Setup training data
        self.room = room
        self.sensors = sensors

        #initialise adjacent room shifting
        self.adj_rooms = room_adj_ls[room]
        shifted_data = data.shift(1)
        shifted_data = shifted_data[self.adj_rooms]
        

        self.vars = [room] + sensors
        self.training_data = data[self.vars]


        # Setup outcome space
        self.outcome_space = self.learn_outcome_space()
        if outcomes_remap is not None:
            self.outcome_space.update(
                {k: v for k, v in outcomes_remap.items() if k in self.outcome_space})

        self.outcome_space[self.room + '_next'] = self.outcome_space[self.room]

        # Setup prediction factors
        self.state_factor = self.learn_states()
        self.transition_factor = self.learn_transitions()
        self.emission_factor = self.learn_emissions()
        self.var_remap = { str(self.room + '_next'): self.room }

        self.hmm = HiddenMarkovModel(
            self.state_factor, self.transition_factor, self.emission_factor, self.var_remap)

    def prediction(self, threshold=0.95, **evidence):
        prediction_factor = self.hmm.forward(normalize=True, **evidence)

        # print(prediction_factor)

        if threshold is not None:
            # TODO: fix this people count index (the '0' is hardcoded)
            if prediction_factor['0'] >= threshold:
                return 'off'
            return 'on'

        mle_index = prediction_factor.table.argmax()
        prediction = self.outcome_space[self.room][mle_index]
        return prediction

    def learn_outcome_space(self) -> dict[str, tuple]:
        '''Learns the outcome space for each variable in the training data'''
        outcome_space = {}
        for column in list(self.training_data.columns):
            values = list(self.training_data[column].unique())
            outcome_space[column] = tuple(values)

        return outcome_space

    def learn_states(self) -> Factor:
        state_factor = Factor((self.room,), self.outcome_space)
        probs = self.training_data[self.room].value_counts(normalize=True)

        for outcome in self.outcome_space[self.room]:
            state_factor[outcome] = probs[outcome] if outcome in probs.index else 0.0

        # print(state_factor.normalize())

        return state_factor

    def learn_transitions(self) -> Factor:
        transition_domain = [self.room, self.room + '_next']
        transition_factor = Factor(tuple(transition_domain), self.outcome_space)

        # Get original and transition column
        room_t0 = self.training_data[self.room]
        room_t1 = room_t0.shift(-1)
        transition_table = pd.concat([room_t0, room_t1], axis=1)
        transition_table.columns = transition_domain

        # Get probabilities
        transition_probs = transition_table.value_counts(normalize=True)
        
        # Copy over transition probabilities
        for transition in permutations(Utils.PEOPLE_COUNT_BUCKETS, 2):
            transition_exists = transition_probs.index.isin([transition]).any()
            probability = transition_probs[transition] if transition_exists else 0.0
            transition_factor[transition[0], transition[1]] = probability
        
        # print(transition_factor.normalize())

        return transition_factor.normalize()

    def learn_emissions(self, alpha=2) -> Factor:
        emission_factor = Utils.estimate_factor(
            self.training_data, self.room, self.sensors, self.outcome_space, alpha)
    
        # print(emission_factor.normalize())

        return emission_factor.normalize()

    