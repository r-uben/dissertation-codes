import numpy as np
from scipy.stats import norm

class Markets(object):

    def __init__(self, T, dt, r):
        self.T  = T
        self.dt = dt
        self.r  = r

    def gbm_sde(self, prev_x, prev_u, ρ, σ):
        dW = np.random.normal(0, 1) * np.sqrt(self.dt)
        return prev_x + σ * prev_u * (ρ * self.dt + dW)

    def gbm_sol(self, prev_x, prev_u, ρ, σ):
        W = np.random.normal(0, 1) * np.sqrt(self.dt)
        D = np.exp(σ * (ρ - σ / 2) * self.dt + σ * W)
        return prev_x + prev_u * (D - 1)

    def crr(self, prev_x, prev_u, u, d):
        p = (np.exp( self.r * self.T - d)) / (u - d)
        ξ = np.random.binomial(1, p)
        D = (d + ξ * (u - d)) * np.exp( -self.r * self.dt)
        return prev_x + prev_u * (D - 1)

    def real_data(self, prev_x, prev_u, df, i):
        S = df['S'].tolist()
        D = S[i+1] / S[i]
        return prev_x + prev_u * (D - 1)
