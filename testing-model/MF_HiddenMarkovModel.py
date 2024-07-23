import numpy as np
from MF_DiscreteFactors import Factor

class HiddenMarkovModel():
    def __init__(self, start_state: Factor, transition: Factor, emission: Factor, variable_remap: dict):
        '''
        Takes 3 arguments:
        - start_state: a factor representing the start state. E.g. domain might be ('A', 'B', 'C')
        - transition: a factor that represents the transition probs. E.g. P('A_next', 'B_next', 'C_next' | 'A', 'B', 'C')
        - emission: emission probabilities. E.g. P('O' | 'A', 'B', 'C')
        - variable_remap: a dictionary that maps new variable names to old variable names,
                            to reset the state after transition. E.g. {'A_next':'A', 'B_next':'B', 'C_next':'C'}
        '''
        self.state = start_state
        self.transition = transition
        self.emission = emission
        self.remap = variable_remap

        self.history = []
        self.prev_history = []

    def forward(self, normalize=False, **emission_evi):
        # get state vars (to be marginalized later)
        state_vars = self.state.domain

        # join with transition factor
        f = self.state*self.transition

        # marginalize out old state vars, leaving only new state vars
        for var in state_vars:
            f = f.marginalize(var)

        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f

        # set emission evidence
        emissionFactor = self.emission.evidence(**emission_evi)

        # join with state factor
        f = self.state*emissionFactor

        # marginalize out emission vars
        for var in f.domain:
            if var not in state_vars:
                f = f.marginalize(var)
        self.state = f

        # normalize state (keep commented out for now)
        if normalize:
            self.state = self.state.normalize()

        return self.state
    
    def forwardBatch(self, n, **emission_evi):
        '''
        emission_evi: A dictionary of lists, each list containing the evidence list for a variable. 
                            Use `None` if no evidence for that timestep
        '''
        history = []
        for i in range(n):
            # select evidence for this timestep
            evi_dict = dict([(key, value[i]) for key, value in emission_evi.items() if value[i] is not None])
            
            # take a step forward
            state = self.forward(**evi_dict)
            history.append(state)
        return history
    
    def viterbi(self, **emission_evi):
        '''
        This function is very similar to the forward algorithm. 
        For simplicity, we will assume that there is only one state variable, and one emission variable.
        '''

        # confirm that state and emission each have 1 variable 
        assert len(self.state.domain) == 1
        assert len(self.emission.domain) == 2
        assert len(self.transition.domain) == 2

        # get state and evidence var names (to be marginalized and maximised out later)
        state_var_name = self.state.domain[0]
        emission_vars = [v for v in self.emission.domain if v not in self.state.domain]
        emission_var_name = emission_vars[0]

        # join with transition factor
        f = self.state*self.transition
        
        # maximize out old state vars, leaving only new state vars
        f, prev = f.maximize(state_var_name, return_prev=True)
        self.prev_history.append(prev)

        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f

        # set emission evidence
        emissionFactor = self.emission.evidence(**emission_evi)

        # join with state factor
        f = self.state*emissionFactor

        # marginalize out emission var if it is in the factor domain
        if emission_var_name in f.domain:
            f = f.marginalize(emission_var_name)

        # normalize state (keep commented out for now)
        # self.state = self.state.normalize()

        self.state = f
        self.history.append(self.state)
        return self.state

    def viterbiBatch(self, n,  **emission_evi):
        '''
        emission_evi: A dictionary of lists, each list containing the evidence list for a variable. 
                         Use `None` if no evidence for that timestep
        '''
        for i in range(n):
            # select evidence for this timestep
            evi_dict = dict([(key, value[i]) for key, value in emission_evi.items() if value[i] is not None])
            self.viterbi(**evi_dict)
        return self.history

    def traceBack(self):
        '''
        This function iterates backwards over the history to find the most 
        likely sequence of states.
        For simplicity, this function assumes there is one state variable
        '''
        # get most likely outcome of final state
        index = np.argmax(self.history[-1].table)
        
        # Go through "prev_history" in reverse
        indexList = []
        for prev in reversed(self.prev_history):
            indexList.append(index)
            index = prev[index]
        indexList = reversed(indexList)

        # translate the indicies into the outcomes they represent
        mleList = []
        stateVar = self.state.domain[0]
        for idx in indexList:
            mleList.append(self.state.outcomeSpace[stateVar][idx]) 
        return mleList