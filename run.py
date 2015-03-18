from __future__ import print_function
import numpy as np
import os, sys

from generation import make_matrices
from simulation import generate_policy, run_episode

dirname = "data/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
attitudes = [('superPessimistic', -5),
             ('pessimistic', -1),
             ('neutral', 0),
             ('optimistic', 1),
             ('superOptimistic', 5)]

N = 5
states, actions, transitions, rewards = make_matrices(N, 2, attitudes, dirname)
horizons = [1, 2, 4, 5, 8, 10, 20, 40]
num_episodes = 100
max_time = 40
neutral_tmat = transitions['neutral']
for horizon in horizons:
    print("Horizon: {}".format(horizon))
    for pair in attitudes:
        attitude = pair[0]
        tmat = transitions[attitude]
        rmat = rewards[attitude]
        policy = generate_policy(horizon, max_time, tmat, rmat)
        episode = run_episode(policy, num_episodes, horizon, max_time, neutral_tmat, rmat)
        totals = np.array([sum(life[:, 0]) for life in episode])
        print("{0}: {1}".format(attitude, np.average(totals)))

