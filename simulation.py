from __future__ import division
import numpy as np
from scipy import stats
import random

def run_episode(policy, num_episodes, horizon, max_time, transitions, rewards, repeating=True):
    """Runs NUM_EPISODES episodes using the policy given and returns the results."""
    deviation_prob = 0.05
    lives = np.zeros([num_episodes, max_time, 4])
    # lives = np.zeros(max_time, 4 * num_episodes)
    num_actions = rewards.shape[0]
    num_states = rewards.shape[1]
    for n in range(num_episodes):
        # or start at random: random.randrange(num_states)
        state = 0
        life = np.zeros([max_time, 4])
        for t in range(max_time):
            action = policy[t, state]
            deviated = random.random() < deviation_prob
            if deviated:
                # perform any other action
                num_actions = transitions.shape[0]
                action = random.choice([x for x in range(num_actions) if x != action])
            life[t, 0] = rewards[action, state]
            life[t, 1] = state
            life[t, 2] = int(deviated)
            life[t, 3] = action
            # rv for next state with distribution from transition probabilites
            next_rv = stats.rv_discrete(name="next_state",
                                         values=(range(num_states),
                                                 transitions[action, state, :]))
            state = next_rv.rvs()
        lives[n] = life
        # lives[:, (n*4):((n+1)*4)] = life
    return lives

def generate_policy(horizon, max_time, transitions, rewards, repeating=True):
    """Generates the policy of our agent, a 2D array with shape MAX_TIME x NUM_STATES
    such that each element is the index of the optimal action at that time and state.
        HORIZON: the planning horizon (positive int)
        MAX_TIME: the number of steps to simulate (positive int)
        TRANSITIONS: an ndarray of transition probabilites, with shape (A, S, S)
        REWARDS: an ndarray of rewards, with shape (A, S)
        REPEATING: boolean, whether or not the policy is repeating
    """
    decision_times = range(0, max_time, horizon)
    num_actions = rewards.shape[0]
    num_states = rewards.shape[1]
    qvals = np.zeros([max_time, num_states, num_actions])
    realvals = np.zeros([max_time, num_states])
    policy = np.zeros([max_time, num_states])
    # beta = 1, so that future rewards aren't discounted
    discount = 1

    # base case: t = 1
    # TODO: make this more general; here, we assume there are only two actions.
    qvals[0, :, 0] = rewards[0]
    qvals[0, :, 1] = rewards[1]
    realvals[0, :] = np.maximum(rewards[0], rewards[1])
    policy[0, rewards[0] == realvals[0, :]] = 0
    policy[0, rewards[1] == realvals[0, :]] = 1

    if horizon == 1:
        # just repeat our step
        realvals[1:max_time, :] = np.tile(realvals[0, :], (max_time - 1, 1))
        policy[1:max_time, :] = np.tile(policy[0, :], (max_time - 1, 1))
    else:
        # compute actions for a horizon, then repeat the policy
        for t in range(1, horizon):
            for s in range(num_states):
                qvals[t, s, 0] = rewards[0, s] * (t + 1)
                qvals[t, s, 1] = rewards[1, s] + (discount * \
                                                  (sum(transitions[1, s, :] * \
                                                       realvals[t - 1, :])))
                realvals[t, :] = np.maximum(qvals[t, :, 0], qvals[t, :, 1])
                policy[t, realvals[t, :] == qvals[t, :, 0]] = 0
                policy[t, realvals[t, :] == qvals[t, :, 1]] = 1
            for k in range(1, max_time // horizon):
                # populate the repeats
                realvals[t + (horizon*k), :] = realvals[t, :]
                policy[t + (horizon*k), :] = policy[t, :]
        for dt in decision_times:
            realvals[dt, :] = realvals[1, :]
            policy[dt, :] = policy[1, :]

    if not repeating:
        # TODO
        pass

    return np.flipud(policy)
