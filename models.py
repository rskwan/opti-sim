from __future__ import division
import numpy as np
from scipy import stats
import math, os

from simulation import generate_policy

# Action and State classes

class Action:
    """A possible action in an MDP.

    Instance variables:
        NAME: the name of this action (string)
        INDEX: the index of this action (nonnegative int)
    """

    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __str__(self):
        return "<Action[{0}]: {1}>".format(self.index, self.name)

class State:
    """A possible state in an MDP.

    Instance variables:
        NAME: the name of this state (str)
        INDEX: the index of this state (nonnegative int)
        IS_FINAL: whether this is a terminating state (boolean)
    """

    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.is_final = False

    def make_final(self):
        self.is_final = True

    def __str__(self):
        return "<State[{0}]: {1}>".format(self.index, self.name)

class Model:
    """An abstract Model to demonstrate the expected API, which only has
    transitions and rewards in general."""

    def __init__(self):
        self.transitions = {}
        self.rewards = {}

class BasicModel:
    """A very basic model with user-provided transition and reward matrices."""

    def __init__(self, attitudes, dirname=None):
        self.transitions = {}
        self.rewards = {}
        self.attitudes = attitudes
        if dirname is not None:
            self.load_from_dir(dirname)

    def load_from_dir(self, dirname):
        attitudes = ['pessimistic', 'neutral', 'optimistic', 'superOptimistic']
        for idx, attitude in enumerate(self.attitudes):
            fname = dirname + 'transitions_{0}_{1}.p'.format(idx, attitude)
            transitions[attitude] = np.load(fname)
            rewards[attitude] = np.load(dirname + 'rewards.p')

class OldModel:
    """Creates a scenario using the model from the original paper, where
    we simply scale the transition probabilities by a power of the reward."""

    def __init__(self, num_states, num_actions, true_tran_func, reward_func,
                 attitudes, dirname):
        """Generate reward and transition matrices using the given parameters
        (states, actions, attitudes) and the biased transition function in the
        old paper, and saves the data as pickled files.
            NUM_STATES: the number of states
            NUM_ACTIONS: the number of actions
            TRUE_TRAN_FUNC: the true transition function ((s0, s1, a) -> probability)
            REWARD_FUNC: the reward function ((s, a) -> nonnegative float)
            ATTITUDES: a list of (string, gamma) pairs
            DIRNAME: directory in which to save data
        """
        self.states = generate_states(range(num_states))
        self.actions = generate_actions(range(num_actions))
        self.transitions = {}
        self.rewards = {}
        self.attitudes = attitudes
        self.dirname = dirname

        self.reward_func = reward_func
        self.true_tran_func = true_tran_func

        self.init_rewards()
        self.init_transitions()

    def init_transitions(self):
        for i, (aname, gamma) in enumerate(self.attitudes):
            transition_func = self.make_transition_func(gamma)
            transitions[aname] = generate_transitions(self.states, self.actions,
                                                      transition_func)
            fname = self.dirname + "transitions_{0}_{1}.p".format(i, aname)
            transitions[aname].dump(fname)

    def make_transition_func(self, gamma):
        """Make biased transition functions, in the style of the paper:
            p_bias(s'|s,a) = p(s'|s,a) * R(s',a)^gamma / Z(s,a)
        where Z(s,a) = sum_i [p(S'=i|s,a) * R(i,a)^gamma]."""
        products = {}
        reward_sums = {}
        for state0 in self.states:
            for action in self.actions:
                total = 0
                for state1 in self.states:
                    r = self.reward_func(state1, action)
                    if r == 0:
                        prod = 0
                    else:
                        prod = self.true_tran_func(state0, state1, action)\
                               * math.pow(r, gamma)
                    total += prod
                    products[(state0.index, state1.index, action.index)] = prod
                reward_sums[(state0.index, action.index)] = total
        def f(state0, state1, action):
            Z = reward_sums[(state0.index, action.index)]
            if Z == 0:
                return self.true_tran_func(state0, state1, action)
            else:
                return products[(state0.index, state1.index, action.index)] / Z

    def init_rewards(self):
        rmat = generate_rewards(self.states, self.actions, self.reward_func)
        for i, (aname, _) in enumerate(self.attitudes):
            rewards[aname] = rmat
            fname = dirname + "rewards_{0}_{1}.p".format(i, aname)
            rewards[aname].dump(fname)

class OptimalValueModel:
    """A model using the optimal value function in place of the reward."""

    def __init__(self, max_time, num_states, num_actions,
                 true_tran_func, reward_func, attitudes, dirname):
        """Generate reward and transition matrices using the given parameters
        (states, actions, attitudes) and the time-based biased transition function
        we propose based on the value function, and saves the data as pickled files.
            MAX_TIME: the maximum number of time steps
            NUM_STATES: the number of states
            NUM_ACTIONS: the number of actions
            TRUE_TRAN_FUNC: the true transition function ((s0, s1, a) -> probability)
            REWARD_FUNC: the reward function ((s, a) -> nonnegative float)
            ATTITUDES: a list of (string, gamma) pairs
            DIRNAME: directory in which to save data
        """
        self.states = generate_states(range(num_states))
        self.actions = generate_actions(range(num_actions))
        self.transitions = {}
        self.rewards = {}
        self.attitudes = attitudes
        self.dirname = dirname

        self.max_time = max_time
        self.reward_func = reward_func
        self.true_tran_func = true_tran_func

        self.init_rewards()
        self.init_transitions()

    def init_transitions(self, horizon):
        neutral_tmat = generate_transitions(self.states, self.actions,
                                            self.true_tran_func)
        realvals, _ = generate_policy(horizon, self.max_time,
                                      neutral_tmat, self.rmat)
        for i, (aname, gamma) in enumerate(self.attitudes):
            transition_func = self.make_transition_func(aname, gamma, realvals)
            transitions[aname] = generate_transitions(self.states, self.actions,
                                                      f, self.max_time)
            fname = dirname + "transitions_{0}_{1}_h{2}.p".format(i, aname, horizon)
            transitions[aname].dump(fname)

    def make_transition_func(self, aname, gamma, realvals):
        """Make a biased transition function using the value function
        (or an approximation), where at each time step,
            p_bias(s'|s,a) = p(s'|s,a) * V(s')^gamma / Z(s,a)
        where Z(s,a) = sum_i [p(S'=i|s,a) * V(i)^gamma].
        These transition functions are _time-based_, so their last
        argument is a timestep.
        """
        products = {}
        value_sums = {}
        for t in range(self.max_time):
            for state0 in self.states:
                for action in self.actions:
                    total = 0
                    for state1 in self.states:
                        v = realvals[t, state1.index]
                        if v == 0:
                            prod = 0
                        else:
                            prod = self.true_tran_func(state0, state1, action)\
                                   * math.pow(v, gamma)
                        total += prod
                        products[(state0.index, state1.index, action.index, t)] = prod
                    value_sums[(state0.index, action.index, t)] = total
        def f(state0, state1, action, time):
            Z = value_sums[(state0.index, action.index, time)]
            if Z == 0:
                return self.true_tran_func(state0, state1, action)
            else:
                return products[(state0.index, state1.index, action.index, time)] / Z
        return f

    def init_rewards(self):
        self.rmat = generate_rewards(self.states, self.actions, self.reward_func)
        for i, (aname, gamma) in enumerate(attitudes):
            rewards[aname] = rmat
            fname = dirname + "rewards_{0}_{1}.p".format(i, aname)
            rewards[aname].dump(fname)

class PMModel:
    """A model for the product management experiment."""

    def __init__(self, num_states, action_vals, final_val, attitudes, dirname):
        """Generate reward and transition matrices using the given parameters
        (states, attitudes), a binomial-based transition function, and a
        normally-distributed reward function, and saves the data as pickled files.
            NUM_STATES: the number of states
            ATTITUDES: a list of (string, scale) vals
            ACTION_VALS: a mapping (index, reward)
            FINAL_VALS: the reward for the final/absorbing state
            DIRNAME: directory in which to save data
        """
        self.states = generate_states(range(num_states))
        self.actions = generate_actions(range(len(action_vals)))
        self.transitions = {}
        self.rewards = {}
        self.attitudes = attitudes
        self.dirname = dirname

        self.action_vals = action_vals
        self.final_val = final_val

        self.init_rewards()
        self.init_transitions()

    def init_transitions(self, horizon=None):
        for i, (aname, scale) in enumerate(attitudes):
            transition_func = self.make_binomial_transition(scale, horizon)
            transitions[aname] = generate_transitions(self.states, self.actions,
                                                      transition_func)
            if horizon is not None:
                fname = dirname + "transitions_{0}_{1}_h{2}.p".format(i, aname, horizon)
            else:
                fname = dirname + "transitions_{0}_{1}.p".format(i, aname)
            transitions[aname].dump()

    def make_binomial_transition(self, scale, horizon):
        """Make transition probabilities using a scaled binomial distribution.
        The actions are "invest" (with index 0) and "market" (with index 1).
        When investing, starting at state S0, the next state is distributed as
            min(100, S0 + Binomial(num_states, p))
        where
            p = scale / (0.8 * horizon)
        This implementation computes the transition probabilties and
        stores them in a lookup table, then returns a closure that serves
        as a transition function, looking up the probability as needed.
        """
        num_states = len(self.states)
        if horizon is not None:
            p = 1 / (0.8 * horizon)
        else:
            p = scale
        rv = stats.binom(num_states, p)
        invest_probs = {}
        for state0 in self.states:
            curr_index = state0.index
            total_minus_final = 0
            for next_index in range(num_states):
                delta = next_index - curr_index
                if delta < 0:
                    transition = 0
                elif next_index != num_states - 1:
                    transition = rv.pmf(delta)
                    total_minus_final += transition
                else:
                    transition = 1 - total_minus_final
                invest_probs[(curr_index, next_index)] = transition
        def f(state0, state1, action):
            if action.index == 0:
                return invest_probs[(state0.index, state1.index)]
            elif action.index == 1:
                return float(state1.index == state0.index)
        return f

    def init_rewards(self):
        num_states = len(self.states)
        def reward_func(state, action):
            if state.index == num_states - 1:
                return self.final_val
            return self.action_vals[action.index]
        rmat = generate_rewards(self.states, self.actions, reward_func)
        for i, (aname, scale) in enumerate(attitudes):
            rewards[aname] = rmat
            fname = dirname + "rewards_{0}_{1}.p".format(i, aname)
            rewards[aname].dump()

# main functions: generate transitions/rewards

def generate_transitions(states, actions, transition_func, max_time=None):
    """Generate transition matrices for all states and actions given using
       TRANSITION_FUNC. Returns a NumPy array with shape (A, S, S) and type float
       if MAX_TIME is None, and shape (T, A, S, S) otherwise.
        STATES: states with indices covering a range [0, S] (iterable of State)
        ACTIONS: actions with indices covering a range [0, A] (iterable of Action)
        TRANSITION_FUNC: function from (state0, state1, action[, time if time-based]) to float.
        MAX_TIME: the number of time steps, if not None, for time-based transitions.
    """
    time_based = max_time is not None
    num_states = len(states)
    num_actions = len(actions)
    if time_based:
        transitions = np.empty([max_time, num_actions, num_states, num_states])
        for t in range(max_time):
            for action in actions:
                for state0 in states:
                    for state1 in states:
                        transitions[t][action.index][state0.index, state1.index] = \
                            transition_func(state0, state1, action, t)
                normalizer = np.tile(transitions[t][action.index].sum(axis=1).\
                                                                  reshape((num_states, 1)),
                                     (1, num_states))
                transitions[t][action.index] /= normalizer
    else:
        transitions = np.empty([num_actions, num_states, num_states])
        for action in actions:
            for state0 in states:
                for state1 in states:
                    transitions[action.index][state0.index, state1.index] = \
                        transition_func(state0, state1, action)
            normalizer = np.tile(transitions[action.index].sum(axis=1).\
                                                           reshape((num_states, 1)),
                                 (1, num_states))
            transitions[action.index] /= normalizer
    return transitions

def generate_rewards(states, actions, reward_func):
    """Generate a reward matrix for all states and actions given using
       REWARD_FUNC. Returns a NumPy array with shape (A, S) and type float.
        STATES: states with indices covering a range [0, S] (iterable of State)
        ACTIONS: actions with indices covering a range [0, A] (iterable of Action)
        REWARD_FUNC: function from (state, action) to float.
    """
    num_states = len(states)
    num_actions = len(actions)
    rewards = np.empty([num_actions, num_states])
    for action in actions:
        for state in states:
            rewards[action.index][state.index] = reward_func(state, action)
    return rewards

# utility functions

def generate_states(names):
    """Return a list of states using the names given."""
    states = [State(str(name), i) for i, name in enumerate(names)]
    states[-1].make_final()
    return states

def generate_actions(names):
    """Return a list of actions using the names given."""
    return [Action(str(name), i) for i, name in enumerate(names)]

# basic transitions and rewards

def deterministic_transition(state0, state1, action):
    if action.index == 0 or state0.is_final:
        # must stay in current state
        return int(state0.index == state1.index)
    else:
        # must quit and go to next state
        return int(state0.index + 1 == state1.index)

def stochastic_transition(state0, state1, action):
    if action.index == 0 or state0.is_final:
        # action is stay; state doesn't change
        return int(state0.index == state1.index)
    else:
        # quit: go to next state with probability p
        p = 0.8
        if state0.index + 1 == state1.index:
            return p
        elif state0.index == state1.index:
            return 1 - p
        else:
            return 0

def basic_reward(state, action):
    if action.index == 0:
        # stay in current state, increase reward linearly
        return state.index + 1
    else:
        # go to next state and suffer
        return 0.1
