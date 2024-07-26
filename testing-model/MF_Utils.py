import numpy as np
import pandas as pd
from itertools import product

from MF_DiscreteFactors import Factor

PEOPLE_COUNT_BUCKETS = ('0', '<3', '<10', '>=10')
TIME_BUCKETS = ('morning', 'afternoon', 'evening')

def bucket_people_count(count: int) -> str:
    if count == 0:
        return PEOPLE_COUNT_BUCKETS[0]
    if count < 3:
        return PEOPLE_COUNT_BUCKETS[1]
    if count < 10:
        return PEOPLE_COUNT_BUCKETS[2]
    return PEOPLE_COUNT_BUCKETS[3]

def _all_equal_this_index(dict_of_arrays: dict[list], **fixed_vars) -> np.ndarray:
    # base index is a boolean vector, everywhere true
    first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
    index = np.ones_like(first_array, dtype=np.bool_)
    for var_name, var_val in fixed_vars.items():
        index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
    return index

def estimate_factor(
        data: pd.DataFrame,
        var_name: str,
        parent_names: list[str],
        outcome_space: dict[str, tuple],
        alpha=1) -> Factor:
    '''Source: Assignment 1 Solution'''
    var_outcomes = outcome_space[var_name]
    parent_outcomes = [outcome_space[var] for var in (parent_names)]
    # cartesian product to generate a table of all possible outcomes
    all_parent_combinations = product(*parent_outcomes)

    f = Factor(list(parent_names)+[var_name], outcome_space)
    
    for i, parent_combination in enumerate(all_parent_combinations):
        parent_vars = dict(zip(parent_names, parent_combination))
        parent_index = _all_equal_this_index(data, **parent_vars)
        for var_outcome in var_outcomes:
            var_index = (np.asarray(data[var_name])==var_outcome)
            f[tuple(list(parent_combination)+[var_outcome])] = ((var_index & parent_index).sum()+alpha)/(parent_index.sum()+alpha*len(var_outcomes))
            
    return f