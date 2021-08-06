import numpy as np
import pandas as pd
import os
import time 
from scipy.stats import norm
from markets import Markets

class EMV(object):

    def __init__(self, d, market, ηw, ηα, ηβ, ηθ, x_0, z, T, dt, λ, M, N, ρ, σ, real_data = pd.DataFrame()):
        # Number of assets
        self.d = d
        # Market
        if real_data.empty is True:
            self.market = market.lower()
        if real_data.empty is False:
            self.market = market
            self.df = real_data
        # Learning rates
        self.ηw     = ηw
        self.ηα     = ηα
        self.ηβ     = ηβ
        self.ηθ     = ηθ
        # Initial wealth
        self.x_0    = x_0
        # Target Payoff
        self.z      = z
        # Investment Horizon
        self.T      = T
        # Discretisation
        self.dt     = dt
        # Exploration Rate
        self.λ      = λ
        # Number of Iterations
        self.M      = M
        # Sample Average Size
        self.N      = N
        # Final Step
        self.final_step = int(np.floor(self.T / self.dt))
        # MARKET (PARAMETERS)
        self.σ      = σ
        self.ρ      = ρ
        # COLLECTED SAMPLES
        self.D      = []
        # INITIAL PARAMETERS
        self.old_α  = np.ones(self.d) * 0.05
        self.old_β  = np.identity(self.d) * 0.05
        self.old_θ0 = 0.05
        self.old_θ1 = 0.05
        self.old_θ2 = 0.05
        self.old_θ3 = 0.05
        self.old_φ  = {
            'α': self.d / 2 * np.inner(self.old_α, self.old_α),
            'β': self.d / 2 * np.log(np.pi * np.e * self.λ) + np.log(np.linalg.det(self.old_β))
        }
        # PARAMETERS TO BE UPDATED
        self.α  = self.old_α
        self.β  = self.old_β
        self.θ0 = self.old_θ0
        self.θ1 = self.old_θ1
        self.θ2 = self.old_θ2
        self.θ3 = self.old_θ3
        self.φ  = self.old_φ
        self.w  = 0
        self.final_wealths      = []
        # DATA FRAMES
        self.episodes           = []
        self.sample_mean        = []
        self.sample_variance    = []
        self.sample_std         = []
        self.annualised_returns = []
        self.θ0_list = [self.θ0]
        self.θ1_list = [self.θ1]
        self.θ2_list = [self.θ2]
        self.θ3_list = [self.θ3]
        self.w_list  = [self.w]
        # Paths
        self.data = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data'

    def _round(self, x):
        return float("{0:.3f}".format(x))

    def _piMean(self, x):
        # Return the mean of the policy (density)
        mean = - np.multiply(self.old_β.dot(self.old_α), (x - self.w))
        return mean

    def _piVariance(self, t):
        first_term = self.λ / 2
        second_term = self.old_β.transpose().dot(self.old_β)
        third_term = np.exp( np.inner(self.old_α, self.old_α) * (self.T - t) )
        # Return the variance of the policy (density)
        variance =   first_term * second_term * third_term
        return variance

    def _nextWealth(self, prev_x, prev_u, i):
        markets = Markets(self.T, self.dt, self.ρ, self.σ, self.d)
        if self.market == 'log':
            next_wealth = markets.gbm_sde(prev_x, prev_u)   
        if self.market == 'exp':
            next_wealth = markets.gbm_sol(prev_x, prev_u)
        if self.market == 'crr':
            next_wealth = markets.crr(prev_x, prev_u)
        if self.market != 'log' and self.market != 'exp' and self.market != 'crr':
            next_wealth = markets.real_data(prev_x, prev_u, self.df, i)
        return next_wealth
    
    def _collectSamples(self):
        # Initial state
        x = self.x_0
        # Initial time
        t = 0
        # Initial sample
        init_sample = [t, x]
        # Collected samples set
        self.D = [init_sample]
        # Sample (t_i, x_i) from Market under πϕ:
        for i in range(self.final_step - 1):
            # Mean and variance
            pi_mean     = self._piMean(x)
            pi_variance = self._piVariance(t)
            # u_i
            u  = np.random.multivariate_normal(pi_mean, pi_variance)
            # t_i
            t  = (i+1) * self.dt
            # x_{t_i}
            x  = self._nextWealth(x, u, i)
            # Collected samples
            self.D.append([t,x])
        # Keep final wealths
        self.final_wealths.append(x)

    def _β(self, i, j):
        βij = np.delete(self.β, (i), axis=0)
        βij = np.delete(βij, (j), axis=1)
        return βij

    def _V(self, t, x):
        first_term  = ((x - self.w) ** 2) * np.exp( - np.inner(self.old_α, self.old_α) * (self.T - t) )
        second_term = self.old_θ2 * t ** 2
        third_term  = self.old_θ1 * t
        fourth_term = self.old_θ0
        V = first_term + second_term + third_term + fourth_term
        return V

    def _dotV(self, i):
        t         = self.D[i][0]
        x         = self.D[i][1]
        next_t    = self.D[i+1][0]
        next_x    = self.D[i+1][1]
        num_diff  = self._V(next_t, next_x) - self._V(t, x)
        dotV      = num_diff / self.dt
        return dotV

    def _H(self, t):
        return self.old_φ['β'] + self.old_φ['α'] * (self.T - t)

    def _TD_error(self, i):
        dotV_i  = self._dotV(i)
        t       = self.D[i][0]
        return (dotV_i - self.λ * self._H(t))
    
    def _partial_αj_φ2(self, j):
        return self.d * self.old_α[j]
    
    def _partial_βij_φ1(self, i, j):
        return np.linalg.det( self._β(i,j) ) / np.linalg.det(self.old_β)

    def _partial_αj_dotV(self, t, x, nextT, nextX, j):
        first_sum  = (nextX - self.w) ** 2 * np.exp( - np.inner(self.old_α, self.old_α) * (self.T - nextT) )
        second_sum = (x - self.w) ** 2 * np.exp( - np.inner(self.old_α, self.old_α) * (self.T - t) )  
        return - 2 * self.old_α[j] * ( first_sum - second_sum) / self.dt

    def _partial_αj_TD_error(self,  t, x, nextT, nextX, j):
        return self._partial_αj_dotV(t, x, nextT, nextX, j) - self.λ * self._partial_αj_φ2(j)
    
    def _partial_βij_TD_error(self, t, i, j):
        return - self.λ * self._partial_βij_φ1(i, j) * (self.T - t)

    def _gradC_α(self):
        grad = np.zeros( self.d )
        for j in range(self.d):
            for i in range(len(self.D)-1):
                t        = self.D[i][0]
                x        = self.D[i][1]
                next_t   = self.D[i+1][0]
                next_x   = self.D[i+1][1]
                grad[j]  = grad[j] + self._TD_error(i) * self._partial_αj_TD_error(t, x, next_t, next_x, j) * self.dt
            
        return grad
    
    def _gradC_β(self):
        grad = np.zeros( (self.d, self.d) )
        for j in range(self.d):
            for k in range(self.d):
                for i in range(len(self.D)-1):
                    t        = self.D[i][0]
                    grad[j, k]  = grad[j, k] + np.multiply(self._TD_error(i) * self.dt, self._partial_βij_TD_error(t, j, k))
        return grad
    
    def _gradC_θ1(self):
        sum = 0
        for i in range(len(self.D)-1):
            sum     += self._TD_error(i) * self.dt
        return sum

    def _gradC_θ2(self):
        sum = 0
        for i in range(len(self.D)-1):
            t         = self.D[i][0]
            next_t    = self.D[i+1][0]
            sum       += self._TD_error(i) * (next_t ** 2 - t ** 2)
        return sum

    def _update_θ0(self):
        self.θ0 =  - self.old_θ2 * self.T * self.T - self.old_θ1 * self.T - (self.w - self.z) * (self.w - self.z)

    def _update_θ1(self):
        self.θ1 = self.old_θ1 - self.ηθ * self._gradC_θ1()

    def _update_θ2(self):
        self.θ2 = self.old_θ2 - self.ηθ * self._gradC_θ2()

    def _update_θ3(self):
        self.θ3 = np.inner(self.α, self.α)

    def _update_α(self):
        self.α = self.old_α - np.multiply(self.ηα, self._gradC_α())
    
    def _update_β(self):
        self.β = self.old_β - np.multiply(-self.ηβ, self._gradC_β())
 
    def _keepDataFrame(self):
        self.θ0_list.append(self.θ0)
        self.θ1_list.append(self.θ1)
        self.θ2_list.append(self.θ2)
        self.θ3_list.append(self.θ3)

    def _oldToNew(self):
        self.old_θ0 = self.θ0
        self.old_θ1 = self.θ1
        self.old_θ2 = self.θ2
        self.old_θ3 = self.θ3
        self.old_α = self.α
        self.old_β = self.β

    def _updateSDA(self):
        # θ1, θ2
        self._update_θ1()
        self._update_θ2()
        # θ0
        self._update_θ0()
        # ϕ1, ϕ2
        self._update_α()
        self._update_β()
        # θ3
        self._update_θ3()

        # data frames
        self._keepDataFrame()
        # old = new
        self._oldToNew()

    def _sampleMean(self, k):
        mean_x = 0
        for j in range(k - self.N, k):
            mean_x += self.final_wealths[j]
        mean_x /= self.N
        return mean_x
    
    def _sampleVar(self, k, mean):
        variance_x = 0
        for j in range(k - self.N, k):
            variance_x += (self.final_wealths[j] - mean) ** 2
        variance_x /= (self.N - 1)
        return variance_x

    def _annualisedReturns(self, k):
        annualised_return_mean = 0
        for j in range(k - self.N, k):
            annualised_return_mean += self.final_wealths[j] / self.x_0 - 1
        annualised_return_mean /= self.N
        return annualised_return_mean

    def _updateW(self, k):
        if k >= self.N:
            ## Sample mean
            mean_x = self._sampleMean(k)
            ## Sample variance
            variance_x = self._sampleVar(k, mean_x)
            ## Annualised returns
            annualised_returns = self._annualisedReturns(k)
            if k % self.N == 0:
                self.episodes.append( k / 50 )
                self.sample_mean.append(mean_x)
                self.sample_variance.append(variance_x)
                self.sample_std.append(np.sqrt(variance_x))
                self.annualised_returns.append(annualised_returns)
            ## Update rule for the Lagrange multiplier, w
            self.w   -= self.ηα * ( mean_x - self.z )
            self.w_list.append(self.w)
            print(k, self._round(self.z), self._round(mean_x), self._round(variance_x), self._round(self.w))

    def _error(self):
        return abs(self.ρ * self.ρ - self.θ3) / self.ρ / self.ρ

    def _giveName(self, name):
        if self.market == 'log' or self.market == 'exp' or self.market == 'crr':
            name = os.path.join(self.data, self.market + '_' + name + '_μ' + str(self.μ) +   '_r' + str(self.r) + '_σ' + str(self.σ) + '_ρ' + str(self._round(self.ρ)) + '_z' + str(self.z) + '_λ' + str(self.λ) + '.csv')
        else:
            name = os.path.join(self.data, self.market.lower() + '_' + name + '_z' + str(self.z) + '.csv') 
        return name 

    def _saveData(self):
        df1 = {'k': self.episodes, 'mean': self.sample_mean, 'variance': self.sample_variance, 'std': self.sample_std, 'μ': self.annualised_returns}
        df2 = {'θ0': self.θ0_list, 'θ1': self.θ1_list,'θ2': self.θ2_list, 'θ3': self.θ3_list, 'ϕ1': self.ϕ1_list, 'ϕ2': self.ϕ2_list}
        df1 = pd.DataFrame(data = df1)
        df1.to_csv(self._giveName('sample_parameters'), sep=',', index=False)
        df2 = pd.DataFrame(data = df2)
        df2.to_csv(self._giveName('rl_parameters'), sep=',', index=False)

    def _printθ3(self, k):
        print(k, self._round(self.θ3))

    def EMV(self):
        # Episodes
        start_time = time.time()
        for k in range(self.M):
            print(k)
            # Collected samples (each try we sample a new 
            self._collectSamples()
            # Descent-Gradient
            self._updateSDA()
            # Update w
            self._updateW(k)
             # Print the evolution of θ3 approximation
            # self._printθ3(k)
        # End for
        # Create a data frame with sample mean and variance
        # self._saveData()
        elapsed_time = time.time() - start_time
        print(str(self._round(elapsed_time)) + 's')