import numpy as np

class Parameters(object):

    def __init__(self, theta, phi, w):
        self.theta  = theta
        self.phi    = phi
        self.w      = w

    def sharpe_ratio(self, vector):
        if len(vector) == 2:
            sharpe_ratio = np.sqrt(2 * self.phi[-1])
        if len(vector) == 3:
            sharpe_ratio = np.sqrt(self.theta[-1])
        return sharpe_ratio

    def variance(self, λ):
        return λ * np.pi * np.e ** (1 - 2 * self.phi[0])
