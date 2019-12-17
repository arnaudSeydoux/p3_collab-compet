import numpy as np
import random
import copy
from collections import namedtuple, deque


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * random.random()
        self.state = x + dx
        return self.state

    def __str__(self):
        return f'OUNoise({self.seed},{self.mu},{self.theta},{self.sigma})'

    def __repr__(self):
        return f'OUNoise({self.seed},{self.mu},{self.theta},{self.sigma})'


class GaussianNoise:
    def __init__(self, size, factor, decay_rate=0.995, min_rate=0.05):
        self.size = size
        self.factor = factor
        self.decay_rate = decay_rate
        self.min_rate = min_rate

    def reset(self):
        self.end()

    def sample(self):
        return np.random.standard_normal(self.size) * self.factor

    def end(self):
        self.factor = max(self.min_rate, self.factor * self.decay_rate)