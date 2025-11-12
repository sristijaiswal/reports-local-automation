import numpy as np

def safe_nan_statistic(func, array, default_value=0):
    """
    Safe wrapper for numpy nan statistical functions that handles all-NaN arrays
    """
    if np.all(np.isnan(array)):
        return default_value
    return func(array)