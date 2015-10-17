from __future__ import division, print_function
import numpy as np

def margined(arr, prop):
    """Returns (min(arr) - epsilon, max(arr) - epsilon), where
    epsilon = (max(arr) - min(arr)) * prop. This gives the range of
    values within arr along with some margin on the ends.
        ARR: a NumPy array
        PROP: a float"""
    worst = arr.min()
    best = arr.max()
    margin = (best - worst) * prop
    return (worst - margin, best + margin)

def margined_pm(arr, err, prop):
    """Does the same thing as margined, except it includes arr +/- err."""
    return margined(np.array([arr + err, arr - err]), prop)

def make_multiprint(filelist):
    """Makes a function that prints its argument to all files in FILELIST."""
    def multiprint(msg):
        for f in filelist:
            print(msg, file=f)
    return multiprint
