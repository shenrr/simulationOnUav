import os
import pickle

import numpy as np


class OrnsteinUhlenbeckActionNoise:
	# 作者取的别人的noise来源
    # from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):

        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(scale=1,size=self.mu.shape)
        self.x_prev = x

        if abs(x[0])>1.5:
            self.x_prev =0
        if abs(x[1])>1.5:
            self.x_prev =0
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def save(self, dir):
        file = os.path.join(dir, 'ounoise.pickle')
        print("I want to load")
        with open(file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dir):
        file = os.path.join(dir, 'ounoise.pickle')
        print("I want to load")
        with open(file, 'rb') as f:
            noise = pickle.load(f)
        return noise