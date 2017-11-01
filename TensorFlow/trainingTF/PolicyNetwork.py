'''
    date: 2017/10/10
    author: scutjason
'''

import numpy as np
import tensorflow as tf
import gym
env = gym.make('CartPole-v0')

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
'''
random_episodes = 0
reword_sum = 0
while random_episodes < 10:
'''