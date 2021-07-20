import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import norm

class EMV(object):

    def __init__(self, α, ηθ, ηϕ, x_0, z, T, dt, λ, M, N, ρ, σ):
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
        self.ρ   = ρ
        self.σ   = σ
        # INITIAL PARAMETERS
        self.old_ϕ1 = 0.01
        self.old_ϕ2 = 0.01
        self.old_θ0 = 0.01
        self.old_θ1 = 0.01
        self.old_θ2 = 0.01
        self.old_θ3 = 0.01
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
        self.final_wealths = []
        # DATA FRAMES
        self.episodes = []
        self.sample_mean = []
        self.sample_variance = []
        self.θ0_list = [self.θ0]
        self.θ1_list = [self.θ1]
        self.θ2_list = [self.θ2]
        self.θ3_list = [self.θ3]
        self.ϕ1_list = [self.ϕ1]
        self.ϕ2_list = [self.ϕ2]
        self.w_list  = [self.w]
        # Paths
        self.data = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data'

    def __pi_mean_(self, x):
        # To clean up the code we just separate the result into different factors
        first_factor    = np.sqrt( (2 * self.ϕ2) / (self.λ * np.pi))
        second_factor   = np.exp( self.ϕ1 - 0.5 )
        coeff           = - first_factor * second_factor
        # Return the mean of the policy (density)
        mean = coeff * (x - self.w)
        return mean

    def __pi_variance_(self, t):
         # To clean up the code we just separate the result into different factors
        num_term = 1 / (2 * np.pi)
        exp_term = np.exp( 2 * self.ϕ2 * (self.T - t) + 2 * self.ϕ1 - 1)
        # Return the variance of the policy (density)
        variance =  num_term * exp_term
        return variance

    def __next_wealth_(self, prev_x, prev_u):
        '''
            This function calculates the (total) wealth of an investor from the previous wealth
            and the current amount invested in the risky asset which has been determined following
            the EMV-algorithm.
        '''
        dW = np.random.normal(0, 1) * np.sqrt(self.dt)
        next_wealth = prev_x + self.σ * prev_u * (self.ρ * self.dt + dW)
        return next_wealth

    def __V_(self, t, x):
        '''
            This function calculates the value function for a certain wealth in a certain moment
        '''
        first_term  = (x - self.w) * (x - self.w) * np.exp( -self.θ3 * (self.T - t))
        second_term = self.θ2 * t * t
        third_term  = self.θ1 * t
        fourth_term = self.θ0
        V = first_term + second_term + third_term + fourth_term
        return V

    def __dotV_(self, D, i):
        '''
            This function calculates the approximate derivative of the value function given two
            consecutives times.
        '''
        t         = D[i][0]
        x         = D[i][1]
        next_t    = D[i+1][0]
        next_x    = D[i+1][1]
        num_diff  = self.__V_(next_t, next_x) - self.__V_(t, x)
        dotV      = num_diff / self.dt
        return dotV

    def __H_(self, t):
        return self.old_ϕ1 + self.old_ϕ2 * (self.T - t)

    def __collect_samples_(self):
        # Initial state
        x = self.x_0
        # Initial time
        t = 0
        # Initial sample
        init_sample = [t, x]
        # Collected samples set
        D = [init_sample]
        # Sample (t_i, x_i) from Market under πϕ:
        for i in range(1, self.final_step + 1):
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
            D.append([t,x])
        self.final_wealths.append(x)
        return D

    def __gradC_θ1_(self, D):
        sum = 0
        for i in range(len(D)-1):
            dotV_i    = self.__dotV_(D, i)
            t_i       = D[i][0]
            sum       += (dotV_i - self.λ * self.__H_(t_i)) * self.dt
        return sum

    def __gradC_θ2_(self, D):
        sum = 0
        for i in range(len(D)-1):
            dotV_i    = self.__dotV_(D, i)
            t_i_plus    = D[i+1][0]
            t_i         = D[i][0]
            sum         += (dotV_i - self.λ * self.__H_(t_i)) * (t_i_plus ** 2 - t_i ** 2)
        return sum

    def __gradC_ϕ1_(self, D):
        sum = 0
        for i in range(len(D)-1):
            dotV_i  = self.__dotV_(D, i)
            t_i       = D[i][0]
            sum       += (dotV_i - self.λ * self.__H_(t_i)) * self.dt
        return - self.λ * sum

    def __gradC_ϕ2_(self, D):
        sum = 0
        for i in range(len(D)-1):
            dotV_i = self.__dotV_(D, i)
            t        = D[i][0]
            x        = D[i][1]
            next_t   = D[i+1][0]
            next_x   = D[i+1][1]
            first_factor          = (dotV_i - self.λ * self.__H_(t)) * self.dt
            first_num_2nd_factor  = (next_x - self.w) * (next_x - self.w) * np.exp( -2 * self.old_ϕ2 * (self.T - next_t) ) * (self.T - next_t)
            second_num_2nd_factor = (x - self.w)      * (x - self.w)      * np.exp( -2 * self.old_ϕ2 * (self.T - t) ) * (self.T - t)
            num_2nd_factor        =  2 * (first_num_2nd_factor - second_num_2nd_factor)
            second_factor         = - num_2nd_factor / self.dt - self.λ * (self.T - t)
            sum += first_factor * second_factor
        return sum

    def __update_θ0_(self):
        return - self.old_θ2 * self.T * self.T - self.old_θ1 * self.T - (self.w - self.z) * (self.w - self.z)

    def __update_θ1_(self, D):
        return self.old_θ1 - self.ηθ * self.__gradC_θ1_(D)

    def __update_θ2_(self, D):
        return self.old_θ2 - self.ηθ * self.__gradC_θ2_(D)

    def __update_θ3_(self):
        return 2 * self.old_ϕ2

    def __update_ϕ1_(self, D):
        return self.old_ϕ1 - self.ηϕ * self.__gradC_ϕ1_( D)

    def __update_ϕ2_(self,  D):
        return self.old_ϕ2 - self.ηϕ * self.__gradC_ϕ2_(D)

    def __update_SDA_(self, D):
        # θ1, θ2
        self.θ1 = self.__update_θ1_(D)
        self.θ1_list.append(self.θ1)
        self.θ2 = self.__update_θ2_(D)
        self.θ2_list.append(self.θ2)
        # ϕ1, ϕ2
        self.ϕ1 = self.__update_ϕ1_(D)
        self.ϕ1_list.append(self.ϕ1)
        self.ϕ2 = self.__update_ϕ2_(D)
        self.ϕ2_list.append(self.ϕ2)
        # Related other parameters
        self.θ0 = self.__update_θ0_()
        self.θ0_list.append(self.θ0)
        self.θ3 = self.__update_θ3_()
        self.θ3_list.append(self.θ3)
        # old = new
        self.old_θ0 = self.θ0
        self.old_θ1 = self.θ1
        self.old_θ2 = self.θ2
        self.old_θ3 = self.θ3
        self.old_ϕ1 = self.ϕ1
        self.old_ϕ2 = self.ϕ2

        # vectors
        self.θ  = [self.θ0, self.θ1, self.θ2, self.θ3]
        self.ϕ  = [self.ϕ1, self.ϕ2]
        return self.θ, self.ϕ

    def __mean_lastN(self, k):
        mean_x = 0
        for j in range(k - self.N, k):
            mean_x += self.final_wealths[j]
        mean_x /= self.N
        return mean_x
    
    def __var_lastN(self, k, mean):
        variance_x = 0
        for j in range(k - self.N, k):
            variance_x += (self.final_wealths[j] - mean) ** 2
        variance_x /= (self.N - 1)
        return variance_x

    def __update_w_(self, k):
        if k >= self.N:
            ## Sample mean
            mean_x = self.__mean_lastN(k)
            ## Sample variance
            variance_x = self.__var_lastN(k, mean_x)
            if k % self.N == 0:
                self.episodes.append( k / 50 )
                self.sample_mean.append(mean_x)
                self.sample_variance.append(variance_x)
            ## Update rule for the Lagrange multiplier, w
            self.w   -= self.α * ( mean_x - self.z )
            self.w_list.append(self.w)
            print(mean_x, self.z, self.w)
        return self.w

    def __error_(self):
        return abs(self.ρ * self.ρ - self.θ3) / self.ρ / self.ρ

    def EMV(self):
        for k in range(self.M):
            # Collected samples (each try we sample a new )
            D = self.__collect_samples_()
            # Descent-Gradient
            self.__update_SDA_(D)
            print(k, self.ρ, np.sqrt(self.θ3), self.__error_())
            # Update w
            self.__update_w_(k)
            # Data frames
        # Create a data frame with sample mean and variance
        print(len(self.episodes), len(self.sample_mean), len(self.sample_variance))
        df1 = {'k': self.episodes, 'mean': self.sample_mean, 'variance': self.sample_variance}
        df2 = {'θ0': self.θ0_list, 'θ1': self.θ1_list,'θ2': self.θ2_list, 'θ3': self.θ3_list, 'ϕ1': self.ϕ1_list, 'ϕ2': self.ϕ2_list}
        df1 = pd.DataFrame(data = df1)
        df1.to_csv(os.path.join(self.data, 'sample_parameters.csv'), sep=';', index=False)
        df2 = pd.DataFrame(data = df2)
        df2.to_csv(os.path.join(self.data, 'rl_parameters.csv'), sep=';', index=False)
        return self.θ, self.ϕ, self.w