import numpy as np
import pandas as pd
import os
import time 
from scipy.stats import norm
from markets import Markets

class EMV(object):

    def __init__(self, market, α, ηθ, ηϕ, x_0, z, T, dt, λ, M, N, μ, σ, r, real_data = pd.DataFrame()):
        # Market
        if real_data.empty is True:
            self.market = market.lower()
        if real_data.empty is False:
            self.market = market
            self.df = real_data
        # Learning rates
        self.α      = α
        self.ηθ     = ηθ
        self.ηϕ     = ηϕ
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
        self.μ      = μ
        self.r      = r
        self.σ      = σ
        self.ρ      = (self.μ - self.r) / self.σ
        # COLLECTED SAMPLES
        self.D      = []
        # INITIAL PARAMETERS
        self.old_ϕ1 = 1
        self.old_ϕ2 = 1
        self.old_θ0 = 1
        self.old_θ1 = 1
        self.old_θ2 = 1
        self.old_θ3 = 1
        # PARAMETERS TO BE UPDATED
        self.ϕ1 = self.old_ϕ1
        self.ϕ2 = self.old_ϕ2
        self.θ0 = self.old_θ0
        self.θ1 = self.old_θ1
        self.θ2 = self.old_θ2
        self.θ3 = self.old_θ3
        self.w  = 1
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
        self.ϕ1_list = [self.ϕ1]
        self.ϕ2_list = [self.ϕ2]
        self.w_list  = [self.w]
        # Paths
        self.data = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data'

    def __round__(self, x):
        return float("{0:.3f}".format(x))

    def __pi_mean__(self, x):
        # To clean up the code we just separate the result into different factors
        first_factor    = np.sqrt( (2 * self.old_ϕ2) / (self.λ * np.pi))
        second_factor   = np.exp( self.old_ϕ1 - 0.5 )
        coeff           = first_factor * second_factor
        # Return the mean of the policy (density)
        mean = - coeff * (x - self.w)
        return mean

    def __pi_variance__(self, t):
         # To clean up the code we just separate the result into different factors
        num_term = 1 / (2 * np.pi)
        exp_term = np.exp( 2 * self.old_ϕ2 * (self.T - t) + 2 * self.old_ϕ1 - 1)
        # Return the variance of the policy (density)
        variance =  num_term * exp_term
        return variance

    def __next_wealth__(self, prev_x, prev_u, i):
        markets = Markets(self.T, self.dt, self. μ, self.r, self.σ)
        if self.market == 'log':
            next_wealth = markets.gbm_sde(prev_x, prev_u)   
        if self.market == 'exp':
            next_wealth = markets.gbm_sol(prev_x, prev_u)
        if self.market == 'crr':
            next_wealth = markets.crr(prev_x, prev_u)
        else:
            next_wealth = markets.real_data(prev_x, prev_u, self.df, i)
        return next_wealth

    def __V__(self, t, x):
        first_term  = (x - self.w) * (x - self.w) * np.exp( -2 * self.old_ϕ2 * (self.T - t))
        second_term = self.old_θ2 * t * t
        third_term  = self.old_θ1 * t
        fourth_term = self.old_θ0
        V = first_term + second_term + third_term + fourth_term
        return V

    def __dotV__(self, i):
        t         = self.D[i][0]
        x         = self.D[i][1]
        next_t    = self.D[i+1][0]
        next_x    = self.D[i+1][1]
        num_diff  = self.__V__(next_t, next_x) - self.__V__(t, x)
        dotV      = num_diff / self.dt
        return dotV

    def __H__(self, t):
        return self.old_ϕ1 + self.old_ϕ2 * (self.T - t)

    def __collect_samples__(self):
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
            pi_mean     = self.__pi_mean__(x)
            pi_variance = self.__pi_variance__(t)
            pi_std      = np.sqrt(pi_variance)
            # u_i
            u  = np.random.normal(pi_mean, pi_std)
            # t_i
            t  = (i+1) * self.dt
            # x_{t_i}
            x  = self.__next_wealth__(x, u, i)
            # Collected samples
            self.D.append([t,x])
        # Keep final wealths
        self.final_wealths.append(x)

    def __gradC_θ1__(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i  = self.__dotV__(i)
            t       = self.D[i][0]
            sum     += (dotV_i - self.λ * self.__H__(t)) * self.dt
        return sum

    def __gradC_θ2__(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i    = self.__dotV__(i)
            t         = self.D[i][0]
            next_t    = self.D[i+1][0]
            sum       += (dotV_i - self.λ * self.__H__(t)) * (next_t ** 2 - t ** 2)
        return sum

    def __gradC_ϕ1__(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i  = self.__dotV__(i)
            t_i     = self.D[i][0]
            sum     += (dotV_i - self.λ * self.__H__(t_i)) * self.dt
        return - self.λ * sum

    def __gradC_ϕ2__(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i = self.__dotV__(i)
            t        = self.D[i][0]
            x        = self.D[i][1]
            next_t   = self.D[i+1][0]
            next_x   = self.D[i+1][1]
            first_factor          = (dotV_i - self.λ * self.__H__(t)) * self.dt
            first_num_2nd_factor  = (next_x - self.w) * (next_x - self.w) * np.exp( -2 * self.old_ϕ2 * (self.T - next_t) ) * (self.T - next_t)
            second_num_2nd_factor = (x - self.w)      * (x - self.w)      * np.exp( -2 * self.old_ϕ2 * (self.T - t) ) * (self.T - t)
            num_2nd_factor        =  2 * (first_num_2nd_factor - second_num_2nd_factor)
            second_factor         = - num_2nd_factor / self.dt - self.λ * (self.T - t)
            sum += first_factor * second_factor
        return sum

    def __update_θ0__(self):
        self.θ0 =  - self.old_θ2 * self.T * self.T - self.old_θ1 * self.T - (self.w - self.z) * (self.w - self.z)

    def __update_θ1__(self):
        self.θ1 = self.old_θ1 - self.ηθ * self.__gradC_θ1__()

    def __update_θ2__(self):
        self.θ2 = self.old_θ2 - self.ηθ * self.__gradC_θ2__()

    def __update_θ3__(self):
        self.θ3 = 2 * self.old_ϕ2

    def __update_ϕ1__(self):
        self.ϕ1 = self.old_ϕ1 - self.ηϕ * self.__gradC_ϕ1__()

    def __update_ϕ2__(self):
        self.ϕ2 = self.old_ϕ2 - self.ηϕ * self.__gradC_ϕ2__()

    def __keep_to_df__(self):
        self.θ0_list.append(self.θ0)
        self.θ1_list.append(self.θ1)
        self.θ2_list.append(self.θ2)
        self.θ3_list.append(self.θ3)
        self.ϕ1_list.append(self.ϕ1)
        self.ϕ2_list.append(self.ϕ2)

    def __old_to_new__(self):
        self.old_θ0 = self.θ0
        self.old_θ1 = self.θ1
        self.old_θ2 = self.θ2
        self.old_θ3 = self.θ3
        self.old_ϕ1 = self.ϕ1
        self.old_ϕ2 = self.ϕ2

    def __update_SDA__(self):
        # θ1, θ2
        self.__update_θ1__()
        self.__update_θ2__()
        # θ0
        self.__update_θ0__()
        # ϕ1, ϕ2
        self.__update_ϕ1__()
        self.__update_ϕ2__()
        # θ3
        self.__update_θ3__()

        # data frames
        self.__keep_to_df__()
        # old = new
        self.__old_to_new__()

    def __mean_lastN__(self, k):
        mean_x = 0
        for j in range(k - self.N, k):
            mean_x += self.final_wealths[j]
        mean_x /= self.N
        return mean_x
    
    def __var_lastN__(self, k, mean):
        variance_x = 0
        for j in range(k - self.N, k):
            variance_x += (self.final_wealths[j] - mean) ** 2
        variance_x /= (self.N - 1)
        return variance_x

    def __annualised_returns__(self, k):
        annualised_return_mean = 0
        for j in range(k - self.N, k):
            annualised_return_mean += self.final_wealths[j] / self.x_0 - 1
        annualised_return_mean /= self.N
        return annualised_return_mean

    def __update_w__(self, k):
        if k >= self.N:
            ## Sample mean
            mean_x = self.__mean_lastN__(k)
            ## Sample variance
            variance_x = self.__var_lastN__(k, mean_x)
            ## Annualised returns
            annualised_returns = self.__annualised_returns__(k)
            if k % self.N == 0:
                self.episodes.append( k / 50 )
                self.sample_mean.append(mean_x)
                self.sample_variance.append(variance_x)
                self.sample_std.append(np.sqrt(variance_x))
                self.annualised_returns.append(annualised_returns)
            ## Update rule for the Lagrange multiplier, w
            self.w   -= self.α * ( mean_x - self.z )
            self.w_list.append(self.w)
            #print(k, self.__round__(self.z), self.__round__(mean_x), self.__round__(variance_x), self.__round__(self.w))

    def __error__(self):
        return abs(self.ρ * self.ρ - self.θ3) / self.ρ / self.ρ

    def __give_name__(self, name):
        if self.market == 'log' or self.market == 'exp' or self.market == 'crr':
            name = os.path.join(self.data, self.market + '_' + name + '_μ' + str(self.μ) +   '_r' + str(self.r) + '_σ' + str(self.σ) + '_ρ' + str(self.__round__(self.ρ)) + '_z' + str(self.z) + '.csv')
        else:
            name = os.path.join(self.data, self.market.lower() + '_' + name + '_z' + str(self.z) + '.csv') 
        return name 

    def __save_data__(self):
        df1 = {'k': self.episodes, 'mean': self.sample_mean, 'variance': self.sample_variance, 'std': self.sample_std, 'μ': self.annualised_returns}
        df2 = {'θ0': self.θ0_list, 'θ1': self.θ1_list,'θ2': self.θ2_list, 'θ3': self.θ3_list, 'ϕ1': self.ϕ1_list, 'ϕ2': self.ϕ2_list}
        df1 = pd.DataFrame(data = df1)
        df1.to_csv(self.__give_name__('sample_parameters'), sep=',', index=False)
        df2 = pd.DataFrame(data = df2)
        df2.to_csv(self.__give_name__('rl_parameters'), sep=',', index=False)

    def __print_θ3__(self, k):
        print(k, self.__round__(self.ρ ** 2), self.__round__(self.θ3), self.__round__(self.θ3 / self.ρ ** 2), self.__round__(self.__error__()))

    def EMV(self):
        # Episodes
        start_time = time.time()
        for k in range(self.M):
            # Collected samples (each try we sample a new 
            self.__collect_samples__()
            # Descent-Gradient
            self.__update_SDA__()
            # Update w
            self.__update_w__(k)
             # Print the evolution of θ3 approximation
            self.__print_θ3__(k)
        # End for
        # Create a data frame with sample mean and variance
        self.__save_data__()
        elapsed_time = time.time() - start_time
        print(str(self.__round__(elapsed_time)) + 's')