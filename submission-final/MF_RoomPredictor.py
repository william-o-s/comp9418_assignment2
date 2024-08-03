'''
    Helper class that defines RoomPredictor objects.
'''

from itertools import permutations

import pandas as pd
import MF_Utils as Utils
from MF_DiscreteFactors import Factor
from MF_HiddenMarkovModel import HiddenMarkovModel

class RoomPredictor:
    '''Helper class to make predictions for each room.'''

    def __init__(
            self,
            data: pd.DataFrame,
            room: str,
            sensors: list[str],
            outcomes_remap: dict[str, tuple] = None) -> None:
        # Setup training data
        self.room = room
        self.sensors = sensors
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
        '''
        Runs a prediction on the next transition. Has an additional side effect of changing internal
        state of the room.
        '''

        prediction_factor = self.hmm.forward(normalize=True, **evidence)

        # prediction is now not off or on, it's bins, so need to return both
        mle_index = prediction_factor.table.argmax()
        prediction = self.outcome_space[self.room][mle_index]
        light = 'on'
        if prediction == '0':
            light = 'off'

        if threshold is not None:
            if prediction_factor['0'] >= threshold:
                return prediction, 'off'
            return prediction, 'on'

        return prediction, light

    def learn_outcome_space(self) -> dict[str, tuple]:
        '''Learns the outcome space for each variable in the training data.'''
        outcome_space = {}
        for column in list(self.training_data.columns):
            values = list(self.training_data[column].unique())
            outcome_space[column] = tuple(values)

        return outcome_space

    def learn_states(self) -> Factor:
        '''Learns the different states for the given room.'''
        state_factor = Factor((self.room,), self.outcome_space)
        probs = self.training_data[self.room].value_counts(normalize=True)

        for outcome in self.outcome_space[self.room]:
            state_factor[outcome] = probs[outcome] if outcome in probs.index else 0.0

        return state_factor

    def learn_transitions(self) -> Factor:
        '''Learns the transition probabilities for the given room.'''
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

        return transition_factor.normalize()

    def learn_emissions(self, alpha=2) -> Factor:
        '''Learns the emission probabilities for the given room.'''
        emission_factor = Utils.estimate_factor(
            self.training_data, self.room, self.sensors, self.outcome_space, alpha)

        return emission_factor.normalize()
    