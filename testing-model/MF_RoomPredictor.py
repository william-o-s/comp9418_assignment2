import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import product
from MF_BayesNet_VE import BayesNet
from MF_DiscreteFactors import Factor
from MF_Graph import Graph
from MF_HiddenMarkovModel import HiddenMarkovModel
from MF_Utils import estimate_factor

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
        prediction_factor = self.hmm.forward(normalize=True, **evidence)

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
        transition_factor['on', 'on'] = transition_probs[('on', 'on')]
        transition_factor['on', 'off'] = transition_probs[('on', 'off')]
        transition_factor['off', 'on'] = transition_probs[('off', 'on')]
        transition_factor['off', 'off'] = transition_probs[('off', 'off')]
        
        return transition_factor

    def learn_emissions(self, alpha=2) -> Factor:
        return estimate_factor(self.training_data, self.room, self.sensors, self.outcome_space, alpha)
    
    