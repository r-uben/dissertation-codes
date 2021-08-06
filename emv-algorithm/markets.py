import numpy as np
from scipy.stats import norm

class Markets(object):

    def __init__(self, T, dt, ρ, σ, d = None, μ = None, r = None):
        self.T  = T
        self.dt = dt
        self.μ  = μ
        self.r  = r
        self.σ  = σ
        if d == 1:
            self.ρ = (self.μ - self.r) / self.σ
            self.up = np.exp( self.σ * np.sqrt(self.dt) )
            self.down = 1 / self.up
            self.p  = ( np.exp( self.r * self.dt ) - self.down ) / ( self.up - self.down )
        else:
            self.ρ = ρ
        self.d  = d 

    def gbm_sde(self, prev_x, prev_u):
        if self.d == 1:
            dW = np.random.normal(0, 1) * np.sqrt(self.dt)
            x  = prev_x + self.σ * prev_u * (self.ρ * self.dt + dW)
        else:
            ε  = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            dW = np.multiply(ε, np.sqrt(self.dt))
            x  = prev_x + np.inner(np.dot(self.σ, prev_u), np.add(np.multiply(self.ρ, self.dt), dW))
        return x

    def gbm_sol(self, prev_x, prev_u):
        if self.d == 1:
            W = np.random.normal(0, 1) * np.sqrt(self.dt)
            D = np.exp(self.σ * (self.ρ - self.σ / 2) * self.dt + self.σ * W)
            x = prev_x + prev_u * (D - 1)
        else:
            W = np.random.normal(0, 1) * np.sqrt(self.dt)
            D = np.exp(self.σ * (self.ρ - self.σ / 2) * self.dt + self.σ * W)
            x = prev_x + prev_u * (D - 1)
        return x

    def crr(self, prev_x, prev_u):
        ξ = np.random.binomial(1, self.p)
        D = (self.down + ξ * (self.up - self.down)) * np.exp( -self.r * self.dt)
        return prev_x + prev_u * (D - 1)

    def real_data(self, prev_x, prev_u, df, i):
        S = df['Close'].tolist()
        D = S[i+1] / S[i]
        return prev_x + prev_u * (D - 1)
