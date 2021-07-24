import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import norm
from markets import Markets

class EMV(object):

    def __init__(self, α, ηθ, ηϕ, x_0, z, T, dt, λ, M, N, μ, σ, r):
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
        self.L= int(np.floor(self.T / self.dt))
        # MARKET (PARAMETERS)
        self.μ      = μ
        self.r      = r
        self.σ      = σ
        self.ρ      = abs(self.μ - self.r) / self.σ
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
        self.θ  = [self.θ0, self.θ1, self.θ2, self.θ3]
        self.ϕ  = [self.ϕ1, self.ϕ2]
        self.w  = 1
        self.D  = []
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

    def __error_(self):
        return abs(self.ρ * self.ρ - self.θ3) / self.ρ / self.ρ

    def __name_(self, name):
        return os.path.join(self.data, name + '_μ' + str(self.μ) +   '_r' + str(self.r) + '_σ' + str(self.σ) + '_ρ' + str(self.ρ) + '.csv')

    def __save_data_(self):
        df1 = {'k': self.episodes, 'mean': self.sample_mean, 'variance': self.sample_variance, 'std': self.sample_std, 'μ': self.annualised_returns}
        df2 = {'θ0': self.θ0_list, 'θ1': self.θ1_list,'θ2': self.θ2_list, 'θ3': self.θ3_list, 'ϕ1': self.ϕ1_list, 'ϕ2': self.ϕ2_list}
        df1 = pd.DataFrame(data = df1)
        df1.to_csv(self.__name_('sample_parameters'), sep=',', index=False)
        df2 = pd.DataFrame(data = df2)
        df2.to_csv(self.__name_('rl_parameters'), sep=',', index=False)

    def __pi_mean_(self, x):
        # To clean up the code we just separate the result into different factors
        first_factor    = np.sqrt( (2 * self.old_ϕ2) / (self.λ * np.pi))
        second_factor   = np.exp( self.old_ϕ1 - 0.5 )
        coeff           = first_factor * second_factor
        # Return the mean of the policy (density)
        mean = - coeff * (x - self.w)
        return mean

    def __pi_variance_(self, t):
         # To clean up the code we just separate the result into different factors
        num_term = 1 / (2 * np.pi)
        exp_term = np.exp( 2 * self.old_ϕ2 * (self.T - t) + 2 * self.old_ϕ1 - 1)
        # Return the variance of the policy (density)
        variance =  num_term * exp_term
        return variance

    def __next_wealth_(self, prev_x, prev_u):
        dW = np.random.normal(0, 1) * np.sqrt(self.dt)
        next_wealth = prev_x + self.σ * prev_u * (self.ρ * self.dt + dW)
        return next_wealth

    def __V_(self, t, x):
        first_term  = (x - self.w) * (x - self.w) * np.exp( -2 * self.old_ϕ2 * (self.T - t))
        second_term = self.old_θ2 * t * t
        third_term  = self.old_θ1 * t
        fourth_term = self.old_θ0
        V = first_term + second_term + third_term + fourth_term
        return V

    def __dotV_(self, i):
        '''
            This function calculates the approximate derivative of the value function given two
            consecutives times.
        '''
        t         = self.D[i][0]
        x         = self.D[i][1]
        next_t    = self.D[i+1][0]
        next_x    = self.D[i+1][1]
        num_diff  = self.__V_(next_t, next_x) - self.__V_(t, x)
        dotV      = num_diff / self.dt
        return dotV

    def __H_(self, t):
        return self.old_ϕ1 + self.old_ϕ2 * (self.T - t)

    def __gradC_θ1_(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i = self.__dotV_(i)
            t      = self.D[i][0]
            sum    += (dotV_i - self.λ * self.__H_(t)) * self.dt
        return sum

    def __gradC_θ2_(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i    = self.__dotV_(i)
            t         = self.D[i][0]
            next_t    = self.D[i+1][0]
            sum       += (dotV_i - self.λ * self.__H_(t)) * (next_t ** 2 - t ** 2)
        return sum

    def __gradC_ϕ1_(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i  = self.__dotV_(i)
            t       = self.D[i][0]
            sum     += (dotV_i - self.λ * self.__H_(t)) * self.dt
        return - self.λ * sum

    def __gradC_ϕ2_(self):
        sum = 0
        for i in range(len(self.D)-1):
            dotV_i = self.__dotV_(i)
            t        = self.D[i][0]
            x        = self.D[i][1]
            next_t   = self.D[i+1][0]
            next_x   = self.D[i+1][1]
            first_factor          = (dotV_i - self.λ  * self.__H_(t)) * self.dt
            first_num_2nd_factor  = (next_x - self.w) * (next_x - self.w) * np.exp( -2 * self.old_ϕ2 * (self.T - next_t) ) * (self.T - next_t)
            second_num_2nd_factor = (x - self.w)      * (x - self.w)      * np.exp( -2 * self.old_ϕ2 * (self.T - t) ) * (self.T - t)
            num_2nd_factor        =  2 * (first_num_2nd_factor - second_num_2nd_factor)
            second_factor         = - num_2nd_factor / self.dt - self.λ * (self.T - t)
            sum += first_factor * second_factor
        return sum

    def __update_θ0_(self):
        self.θ0 = - self.θ2 * self.T ** 2 - self.θ1 * self.T - (self.w - self.z) ** 2

    def __update_θ1_(self):
        self.θ1 -= self.ηθ * self.__gradC_θ1_()

    def __update_θ2_(self):
        self.θ2 -= self.ηθ * self.__gradC_θ2_()

    def __update_θ3_(self):
        self.θ3 =  2 * self.ϕ2

    def __update_ϕ1_(self):
        self.ϕ1 -= self.ηϕ * self.__gradC_ϕ1_()

    def __update_ϕ2_(self):
        self.ϕ2 -= self.ηϕ * self.__gradC_ϕ2_()

    def __update_vectors_(self):
        self.θ  = [self.θ0, self.θ1, self.θ2, self.θ3]
        self.ϕ  = [self.ϕ1, self.ϕ2]

    def __acumulate_parameters_(self):
        self.θ0_list.append(self.θ0)
        self.θ1_list.append(self.θ1)
        self.θ2_list.append(self.θ2)
        self.θ3_list.append(self.θ3)
        self.ϕ1_list.append(self.ϕ1)
        self.ϕ2_list.append(self.ϕ2)

    def __update_SDA_(self):
        # θ1, θ2
        self.__update_θ1_()
        self.__update_θ2_()
        # θ0
        self.__update_θ0_()
        # ϕ1, ϕ2
        self.__update_ϕ1_()
        self.__update_ϕ2_()
        # θ3
        self.__update_θ3_()
        # keep in lists
        self.__acumulate_parameters_()
        # vectors
        self.__update_vectors_()

    def __mean_lastN_(self, k):
        mean_x = 0
        for j in range(k - self.N, k):
            mean_x += self.final_wealths[j]
        mean_x /= self.N
        return mean_x
    
    def __var_lastN_(self, k, mean):
        variance_x = 0
        for j in range(k - self.N, k):
            variance_x += (self.final_wealths[j] - mean) ** 2
        variance_x /= (self.N - 1)
        return variance_x

    def __annualised_returns_(self, k):
        annualised_return_mean = 0
        for j in range(k - self.N, k):
            annualised_return_mean += self.final_wealths[j] / self.x_0 - 1
        annualised_return_mean /= self.N
        return annualised_return_mean

    def __update_w_(self, k):
        if k >= self.N:
            ## Sample mean
            mean_x = self.__mean_lastN_(k)
            ## Sample variance
            variance_x = self.__var_lastN_(k, mean_x)
            ## Annualised returns
            annualised_returns = self.__annualised_returns_(k)
            if k % self.N == 0:
                self.episodes.append( k / 50 )
                self.sample_mean.append(mean_x)
                self.sample_variance.append(variance_x)
                self.sample_std.append(np.sqrt(variance_x))
                self.annualised_returns.append(annualised_returns)
            ## Update rule for the Lagrange multiplier, w
            self.w   -= self.α * ( mean_x - self.z )
            self.w_list.append(self.w)
        return self.w
    
    def __update_π_(self):
        self.old_ϕ1 = self.ϕ1
        self.old_ϕ2 = self.ϕ2
        self.old_θ0 = self.θ0
        self.old_θ1 = self.θ1
        self.old_θ2 = self.θ2
        self.old_θ3 = self.θ3

    def __collect_samples_(self):
        # Initial state
        x = self.x_0
        # Initial time
        t = 0
        # Initial sample
        init_sample = [t, x]
        # Collected samples set
        self.D = [init_sample]
        # Sample (t_i, x_i) from Market under πϕ:
        for i in range(1, self.L + 1):
            # Mean and variance
            pi_mean     = self.__pi_mean_(x)
            pi_variance = self.__pi_variance_(t)
            pi_std      = np.sqrt(pi_variance)
            # u_i
            u  = np.random.normal(pi_mean, pi_std)
            # t_i
            t  = i * self.dt
            # x_{t_i}
            x  = self.__next_wealth_(x, u)
            # Collected samples
            self.D.append([t,x])
        # Save final wealth
        self.final_wealths.append(x)
    
    def __episode_(self):
        # Initial state
        x = self.x_0
        # Initial time
        t = 0
        # Initial sample
        init_sample = [t, x]
        # Collected samples set
        self.D = [init_sample]
        for i in range(self.L):
            # Mean and variance
            pi_mean     = self.__pi_mean_(x)
            pi_variance = self.__pi_variance_(t)
            pi_std      = np.sqrt(pi_variance)
            # u_i
            u  = np.random.normal(pi_mean, pi_std)
            # t_i
            t  = i * self.dt
            # x_{t_i}
            x  = self.__next_wealth_(x, u)
            self.D.append([t,x])
            # Update
            self.__update_SDA_()
        # Save final_wealths
        self.final_wealths.append(x)

    def EMV(self):
        for k in range(self.M):
            # Policy Evaluation
            self.__episode_()
            # Policy Improvement
            self.__update_π_()
            print(k, self.ρ, np.sqrt(self.θ3), self.__error_())
            # Update Lagrange Multiplier
            self.__update_w_(k)
        # Create a data frame with sample mean and variance
        self.__save_data_()
        return self.θ, self.ϕ, self.w