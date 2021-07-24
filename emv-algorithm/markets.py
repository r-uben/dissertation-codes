import numpy as np
from scipy.stats import norm

class Markets(objects):

    def __init__(self, T, dt, r):
        self.T  = T
        self.dt = dt
        self.r  = r

    def gbm_sde(self, prev_x, prev_u, ρ, σ):
        dW = np.random.normal(0, 1) * np.sqrt(self.dt)
        return prev_x + self.σ * prev_u * (self.ρ * self.dt + dW)

    def gbm_sol(self, prev_x, prev_u, ρ, σ):
        W = np.random.normal(0, 1) * np.sqrt(self.dt)
        D = np.exp((ρ - 0.5 * σ) * self.dt + W)
        return prev_x + prev_u * (D - 1)

    def crr(self, prev_x, prev_u, u, d):
        p = (np.exp( self.r * self.T - d)) / (u - d)
        ξ = np.random.binomial(1, p)
        D = (d + ξ * (u - d)) * np.exp( -self.r * self.dt)
        return prev_x + prev_u * (D - 1)


