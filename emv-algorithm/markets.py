import numpy as np
from scipy.stats import norm

class Markets(object):

    def __init__(self, T, dt, μ, r, σ):
        self.T  = T
        self.dt = dt
        self.μ  = μ
        self.r  = r
        self.σ  = σ
        self.ρ  = (self.μ - self.r) / self.σ
        self.up = np.exp( self.σ * np.sqrt(self.dt) )
        self.down = 1 / self.up
        self.p  = ( np.exp( self.r * self.dt ) - self.down ) / ( self.up - self.down ) 

    def gbm_sde(self, prev_x, prev_u):
        dW = np.random.normal(0, 1) * np.sqrt(self.dt)
        return prev_x + self.σ * prev_u * (self.ρ * self.dt + dW)

    def gbm_sol(self, prev_x, prev_u):
        W = np.random.normal(0, 1) * np.sqrt(self.dt)
        D = np.exp(self.σ * (self.ρ - self.σ / 2) * self.dt + self.σ * W)
        return prev_x + prev_u * (D - 1)

    def crr(self, prev_x, prev_u):
        ξ = np.random.binomial(1, self.p)
        D = (self.down + ξ * (self.up - self.down)) * np.exp( -self.r * self.dt)
        return prev_x + prev_u * (D - 1)

    def real_data(self, prev_x, prev_u, df, i):
        S = df['Close'].tolist()
        D = S[i+1] / S[i]
        return prev_x + prev_u * (D - 1)
