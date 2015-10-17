from __future__ import division, print_function
import numpy as np
import argparse
import csv
import math
import os
import cPickle

from generation import make_matrices_old, make_matrices_valopt, make_matrices_pmexp,\
                       basic_reward, stochastic_transition
from simulation import generate_policy, run_episode
from plots import scenario_plot, make_existing_plot
from util import margined, margined_pm, make_multiprint

# TODO: refactor this function
def run_scenario(scenario, num_episodes):
    """Runs the desired scenario with the specified number of episodes.
       Returns a list of attitudes, a list of horizons, and a NumPy array RESULTS
       with shape (len(attitudes), len(horizons), 3),
       such that RESULTS[attitude_idx][horizon_idx] consists of
       [horizon, average earnings, standard error].
        SCENARIO: a two-element list, with the first specifying the data
                  (e.g. "sequence") and the second specifying the type (e.g. "old")
        NUM_EPISODES: a positive integer for the number of episodes to run
    """
    dirname = "data/{}/".format("_".join(scenario))

    # setup
    if scenario[0] == "sequence":
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        attitude_pairs = [('megaPessimistic', -50),
                          ('ultraPessimistic', -20),
                          ('superPessimistic', -5),
                          ('pessimistic', -1),
                          ('neutral', 0),
                          ('optimistic', 1),
                          ('superOptimistic', 5),
                          ('ultraOptimistic', 20),
                          ('megaOptimistic', 50)]
        attitudes = [p[0] for p in attitude_pairs]
        num_states = 20
        num_actions = 2
        true_tran_func = stochastic_transition
        reward_func = basic_reward
        if scenario[1] == "old":
            states, actions, transitions, rewards = \
                make_matrices_old(num_states, num_actions,
                                  true_tran_func, reward_func,
                                  attitude_pairs, dirname)
    elif scenario[0] == "gen_from_fixed":
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        attitude_pairs = [('ultraPessimistic', -20),
                          ('superPessimistic', -5),
                          ('pessimistic', -1),
                          ('neutral', 0),
                          ('optimistic', 1),
                          ('superOptimistic', 5),
                          ('ultraOptimistic', 20)]
        attitudes = [p[0] for p in attitude_pairs]

        true_transitions = np.load('data/fixed_old/transitions_1_neutral.p')
        rmat = np.load('data/fixed_old/rewards.p')
        num_states = true_transitions.shape[1]
        num_actions = true_transitions.shape[0]
        true_tran_func = lambda s0, s1, a: true_transitions[a.index][s0.index][s1.index]
        reward_func = lambda s, a: rmat[a.index][s.index]
        if scenario[1] == "old":
            states, actions, transitions, rewards = \
                make_matrices_old(num_states, num_actions,
                                  true_tran_func, reward_func,
                                  attitude_pairs, dirname)
    elif scenario == ["fixed", "old"]:
        attitudes = ['pessimistic', 'neutral', 'optimistic', 'superOptimistic']
        transitions = {}
        rewards = {}
        for idx, attitude in enumerate(attitudes):
            transitions[attitude] = np.load(dirname +
                                            'transitions_{0}_{1}.p'.format(idx, attitude))
            rewards[attitude] = np.load(dirname + 'rewards.p')
    elif scenario[0] == "pmexp":
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        attitude_pairs = [('pessimistic', 0.05),
                          ('neutral', 0.07),
                          ('optimistic', 0.15)]
        attitudes = [p[0] for p in attitude_pairs]
        num_states = 100
        action_vals = (-15000, 10000)
        final_val = 135000
        states, actions, transitions, rewards = \
            make_matrices_pmexp(num_states, action_vals, final_val,
                                attitude_pairs, dirname)
    else:
        raise Exception("Invalid scenario. Modify and try again.")

    csvout = open(dirname + "results_{0}_data.csv".format(num_episodes), 'w+')
    writer = csv.writer(csvout)
    writer.writerow(["Horizon", "AvgEarnings", "StdErr", "AttitudeIndex", "AttitudeName"])
    readableout = open(dirname + "results_{0}_readable.txt".format(num_episodes), 'w+')
    verboseout = open(dirname + "results_{0}_verbose.txt".format(num_episodes), 'w+')
    print_readverb = make_multiprint([readableout, verboseout])

    horizons = [1, 2, 5, 10, 20, 40, 60, 80]
    if scenario[0] == "pmexp":
        horizons = [4, 12, 24, 72]
    max_time = max(horizons)
    results = np.zeros((len(attitudes), len(horizons), 3))
    time_based = False
    if scenario[1] == "valopt":
        time_based = True
    # note: "neutral" must be an attitude
    for horizon_idx, horizon in enumerate(horizons):
        print_readverb("Horizon: {}".format(horizon))
        if scenario[1] == "valopt":
            states, actions, transitions, rewards = \
                make_matrices_valopt(horizon, max_time, num_states, num_actions,
                                     true_tran_func, reward_func, attitude_pairs, dirname)
        neutral_tmat = transitions['neutral']
        for attitude_idx, attitude in enumerate(attitudes):
            tmat = transitions[attitude]
            rmat = rewards[attitude]
            _, policy = generate_policy(horizon, max_time, tmat, rmat, time_based)
            episode = run_episode(policy, num_episodes, max_time,
                                  neutral_tmat, rmat, time_based)
            totals = np.array([sum(life[:, 0]) for life in episode])
            mean = np.average(totals)
            stderr = np.std(totals) / math.sqrt(num_episodes)
            results[attitude_idx][horizon_idx] = [horizon, mean, stderr]
            writer.writerow([horizon, mean, stderr, attitude_idx, attitude])
            print_readverb("{0}: mean = {1}, stderr = {2}".format(attitude, mean, stderr))
            print("Totals: {0}".format(totals), file=verboseout)
            print("Policy for {0}: {1}".format(attitude, policy), file=verboseout)
            polfname = "policy_{0}_{1}_{2}_{3}.p".format(num_episodes, horizon, attitude_idx, attitude)
            cPickle.dump((attitude, policy), open(dirname + polfname, 'w+'))

    scenario_plot(results, attitudes, horizons, num_episodes,
                  dirname + "results_{0}_plot.png".format(num_episodes))

    return results, attitudes, horizons

def arg_setup():
    """Sets up an argument parser and returns the arguments."""
    parser = argparse.ArgumentParser(
             description="run simulations of agents with wishful thinking")
    parser.add_argument('scenario',
                        help="""the scenario to use (currently: sequence, fixed,
                        gen_from_fixed, pmexp)""")
    parser.add_argument('model', help="the kind of model to use (currently: old, valopt)")
    parser.add_argument('episodes', type=int, help="the number of episodes to run")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_setup()
    run_scenario((args.scenario, args.model), args.episodes)
