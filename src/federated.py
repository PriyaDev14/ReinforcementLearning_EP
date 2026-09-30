import numpy as np

def federated_average(q_tables):
    states = set()
    for table in q_tables:
        states.update(table.keys())

    result = {}
    for state in states:
        arrays = [table[state] for table in q_tables if state in table]
        if arrays:
            result[state] = np.mean(arrays, axis=0)
    return result

def copy_model(model):
    return {k: np.array(v, copy=True) for k, v in model.items()}
