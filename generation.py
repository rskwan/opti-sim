from __future__ import division
import numpy as np
import math, os

import mdptoolbox

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

# main functions: generate transitions/rewards

def generate_transitions(states, actions, transition_func):
    """Generate transition matrices for all states and actions given using
       TRANSITION_FUNC. Returns a NumPy array with shape (A, S, S) and type float.
        STATES: states with indices covering a range [0, S] (iterable of State)
        ACTIONS: actions with indices covering a range [0, A] (iterable of Action)
        TRANSITION_FUNC: function from (state0, state1, action) to float.
    """
    num_states = len(states)
    num_actions = len(actions)
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

def make_biased_transition_old(gamma, reward, actions):
    """Make a biased transition function, in the style of the paper.
    Not normalized.
    GAMMA is the parameter representing how optimistic (positive)
    or pessimistic (negative) our agent is. REWARD is a function
    taking (state, action) to some nonnegative float. ACTIONS
    is a list of actions."""
    def f(state0, state1, action):
        max_reward = max([reward(state1, a) for a in actions])
        return stochastic_transition(state0, state1, action) * \
               math.pow(max_reward, gamma)
    return f

def basic_reward(state, action):
    if action.index == 0:
        # stay in current state, increase reward linearly
        return state.index + 1
    else:
        # go to next state and suffer
        return 0.1

# put it all together

def make_matrices(num_states, num_actions, attitudes, dirname):
    """Generate reward and transition matrices using the given parameters
    (states, actions, attitudes), and saves the data
    to DIRNAME as pickled files. ATTITUDES is a list of (string, gamma)
    pairs."""
    states = generate_states(range(num_states))
    actions = generate_actions(range(num_actions))
    transitions = {}
    rewards = {}
    for i, (aname, gamma) in enumerate(attitudes):
        transition_func = make_biased_transition_old(gamma, basic_reward, actions)
        transitions[aname] = generate_transitions(states, actions, transition_func)
        rewards[aname] = generate_rewards(states, actions, basic_reward)
        transitions[aname].dump(dirname + "transitions_{0}_{1}.p".format(i, aname))
        rewards[aname].dump(dirname + "rewards_{0}_{1}.p".format(i, aname))
    return states, actions, transitions, rewards
