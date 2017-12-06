#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import random
from collections import deque

import numpy as np

import gym


class TabularController(object):

    actions_space = range(2)

    def __init__(self, gamma=1., minvisits=10, minvar=1.e-10):
        # discount factor
        self.gamma = gamma
        # min number of visits per state-action
        self.minvisits = minvisits
        # minimum variance per state-action, protection in case of deterministic rewards
        self.minvar = minvar
        # Q(state, action) sample mean value
        self.Qmean = np.zeros(
            (5, len(self.actions_space))
            ) # 5 state-rows, 2 action-columns
        # number of visits per state-action
        self.visits = np.ones(self.Qmean.shape)
        # sum of squared future rewards
        self.Sr2 = np.zeros(self.Qmean.shape)
        # Q(state, action) sample variance
        self.Qvar = np.ones(self.Qmean.shape) * np.inf

    def optimal_action(self, state):
        visits = self.visits[state, :]
        if np.min(visits) < self.minvisits:
            return np.random.choice(self.actions_space)
        else:
            means, variances = self.Qmean[state, :], self.Qvar[state, :]
            sigmas = np.sqrt(np.divide(variances, visits))
            draws = means + sigmas * np.random.randn(len(self.actions_space))
            return self.action_from_values(draws)

    def action_from_values(self, values):
        maxQstate = np.max(values)
        possible_actions = [a for a in self.actions_space if values[a] >= maxQstate]
        return np.random.choice(possible_actions)

    def train(self, state, action, reward, next_state):
        future_reward = reward + self.gamma * np.max(self.Qmean[next_state, :])
        # update mean, sum squared rewards and variance in exact order
        self.update_mean(state, action, future_reward)
        self.update_sum_squared_rewards(state, action, future_reward)
        self.update_variance(state, action)
        self.visits[state, action] += 1

    def update_mean(self, state, action, future_reward):
        visits = self.visits[state, action]
        Qm = self.Qmean[state, action]
        self.Qmean[state, action] += (future_reward - Qm) / visits

    def update_sum_squared_rewards(self, state, action, future_reward):
        self.Sr2[state, action] += future_reward * future_reward

    def update_variance(self, state, action):
        visits = self.visits[state, action]
        if visits > 1:
            sr2 = self.Sr2[state, action]
            Qm = self.Qmean[state, action]
            self.Qvar[state, action] = max(
                (sr2 - visits * Qm * Qm) / (visits - 1),
                self.minvar
            )


def play(episodes=100, benchmark=1900, verbose=False):
    env = gym.make("NChain-v0")

    Qtable = TabularController(gamma=1.0, minvisits=10)

    performance = deque(maxlen=100)
    performance.append(0.)

    episode = 0
    while episode < episodes and np.mean(performance) < benchmark:
        episode += 1
        state = env.reset()

        steps, rewards, done = 0, [], False
        while not done:
            steps += 1
            action = Qtable.optimal_action(state)
            next_state, reward, done, _ = env.step(action)
            Qtable.train(state, action, reward, next_state)
            rewards.append(reward)
            state = next_state
        performance.append(np.sum(rewards))
        if verbose:
            print("episode {} steps {} rewards {} total {}".format(episode, steps, rewards, np.sum(rewards)))

    return episode


if __name__ == '__main__':

    episodes = 200
    benchmark = 1900
    nplays = 100
    results = np.array([play(episodes, benchmark, verbose=False) for _ in range(nplays)])
    success = results < episodes
    print(
        "Total number of successful plays to reach benchmark {} is {}/{}".format(
            benchmark, np.sum(success), nplays
        )
    )
    print("Average number of episodes before success per play {}".format(np.mean(results[success])))
