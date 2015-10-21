from __future__ import division, print_function
import numpy as np
import argparse
import csv
import math
import os
import cPickle

from models import BasicModel, OldModel, OptimalValueModel, PMModel,\
                   basic_reward, stochastic_transition
from simulation import const_policy, generate_policy, run_episode
from plots import scenario_plot
from util import make_multiprint

def run_scenario(scenario, model_type, num_episodes):
    """Runs the desired scenario with the specified number of episodes.
       Returns a list of attitudes, a list of horizons, and a NumPy array RESULTS
       with shape (len(attitudes), len(horizons), 3),
       such that RESULTS[attitude_idx][horizon_idx] consists of
       [horizon, average earnings, standard error].
        SCENARIO: a two-element list, with the first specifying the data
                  (e.g. "sequence") and the second specifying the type (e.g. "old")
        NUM_EPISODES: a positive integer for the number of episodes to run
    """
    dirname = "data/{}/".format("_".join([scenario, model_type]))
    horizons = [1, 2, 5, 10, 20, 40, 60, 80]
    if scenario == "pmexp":
        horizons = [4, 12, 24, 72]
    max_time = max(horizons)
    # init attitudes and model
    if scenario == "sequence":
        attitudes, model = make_sequence_model(model_type, dirname, max_time)
    elif scenario == "gen_from_fixed":
        attitudes, model = make_from_fixed(model_type, dirname)
    elif scenario == "pmexp":
        attitudes, model = make_pm_model(dirname)
    elif scenario == "fixed":
        attitudes = ['pessimistic', 'neutral', 'optimistic', 'superOptimistic']
        model = BasicModel(attitudes, dirname)
    else:
        raise Exception("Invalid scenario. Modify and try again.")
    results = run_sims(dirname, horizons, attitudes, model_type, model, num_episodes)
    scenario_plot(results, attitudes, horizons, num_episodes,
                  dirname + "results_{0}_plot.png".format(num_episodes))
    return results, attitudes, horizons

def run_sims(dirname, horizons, attitudes, model_type, model, num_episodes):
    max_time = max(horizons)
    time_based = False
    if model_type == "valopt":
        time_based = True
    # output setup
    csvout = open(dirname + "results_{0}_data.csv".format(num_episodes), 'w+')
    writer = csv.writer(csvout)
    writer.writerow(["Horizon", "AvgEarnings", "StdErr", "AttitudeIndex", "AttitudeName"])
    readableout = open(dirname + "results_{0}_readable.txt".format(num_episodes), 'w+')
    verboseout = open(dirname + "results_{0}_verbose.txt".format(num_episodes), 'w+')
    print_readverb = make_multiprint([readableout, verboseout])
    # note: "neutral" must be an attitude
    results = np.zeros((len(attitudes), len(horizons), 3))
    for horizon_idx, horizon in enumerate(horizons):
        print_readverb("Horizon: {}".format(horizon))
        if model_type == "valopt":
            model.init_transitions(horizon)
        neutral_tmat = model.transitions['neutral']
        for attitude_idx, attitude in enumerate(attitudes):
            # compute results
            tmat = model.transitions[attitude]
            rmat = model.rewards[attitude]
            if model_type == "const":
                # TODO: allow const to have something other than 0
                policy = const_policy(0, max_time, rmat.shape[1])
            else:
                _, policy = generate_policy(horizon, max_time, tmat, rmat, time_based)
            episode = run_episode(policy, num_episodes, max_time,
                                  neutral_tmat, rmat, time_based)
            totals = np.array([sum(life[:, 0]) for life in episode])
            mean = np.average(totals)
            stderr = np.std(totals) / math.sqrt(num_episodes)
            results[attitude_idx][horizon_idx] = [horizon, mean, stderr]
            # output to files
            writer.writerow([horizon, mean, stderr, attitude_idx, attitude])
            print_readverb("{0}: mean = {1}, stderr = {2}".format(attitude, mean, stderr))
            print("Totals: {0}".format(totals), file=verboseout)
            print("Policy for {0}: {1}".format(attitude, policy), file=verboseout)
            polfname = "policy_{0}_{1}_{2}_{3}.p".format(num_episodes, horizon,
                                                         attitude_idx, attitude)
            cPickle.dump((attitude, policy), open(dirname + polfname, 'w+'))
    return results

def make_sequence_model(model_type, dirname, max_time):
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
    if model_type == "old":
        return attitudes, OldModel(num_states, num_actions,
                                   true_tran_func, reward_func,
                                   attitude_pairs, dirname)
    elif model_type == "valopt":
        return attitudes, OptimalValueModel(max_time, num_states, num_actions,
                                            true_tran_func, reward_func,
                                            attitude_pairs, dirname)
    else:
        raise Exception("Improper model argument. Try again!")

def make_from_fixed(model_type, dirname):
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
    if model_type == "old":
        return attitudes, OldModel(num_states, num_actions,
                                   true_tran_func, reward_func,
                                   attitude_pairs, dirname)
    elif model_type == "valopt":
        return attitudes, OptimalValueModel(max_time, num_states, num_actions,
                                            true_tran_func, reward_func,
                                            attitude_pairs, dirname)
    else:
        raise Exception("Improper model argument. Try again!")

def make_from_fixed(model_type, dirname):
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

def make_pm_model(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # TODO: allow user to put in different parameters here
    attitude_pairs = [('pessimistic', 0.05),
                      ('neutral', 0.07),
                      ('optimistic', 0.15)]
    attitudes = [p[0] for p in attitude_pairs]
    num_states = 100
    action_vals = (-15000, 10000)
    final_val = 135000
    return attitudes, PMModel(num_states, action_vals, final_val,
                              attitude_pairs, dirname)

def arg_setup():
    """Sets up an argument parser and returns the arguments."""
    parser = argparse.ArgumentParser(
             description="run simulations of agents with wishful thinking")
    parser.add_argument('scenario',
                        help="""the scenario to use (currently: sequence, fixed,
                        gen_from_fixed, pmexp)""")
    parser.add_argument('model', help="the kind of model to use (currently: old, valopt, const)")
    parser.add_argument('episodes', type=int, help="the number of episodes to run")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_setup()
    run_scenario(args.scenario, args.model, args.episodes)
