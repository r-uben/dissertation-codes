import numpy as np
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

    def __take_row_(self, D, k):
        D_k = []
        for key in D:
            if key == k:
                D_k.append(D[key])
            else:
                print('There is not such a key')
        return D_k

    def __pi_mean_(self, x):
        # To clean up the code we just separate the result into different factors
        first_factor    = np.sqrt( (2 * self.ϕ2) / (self.λ * np.pi))
        second_factor   = np.exp( (2 * self.ϕ1 - 1) / 2) 
        coeff           = - first_factor * second_factor
        # Return the mean of the policy (density)
        mean = coeff * (x - self.w)
        return mean

    def __pi_variance_(self, t):
         # To clean up the code we just separate the result into different factors
        num_term = 1 / (2 * np.pi)
        exp_term = np.exp(2 * self.ϕ2 * (self.T - t) + 2 * self.ϕ1 - 1)
        # Return the variance of the policy (density)
        variance =  num_term * exp_term
        return variance

    def __V_(self, t, x):
        ''' 
            This function calculates the value function for a certain wealth in a certain moment    
        '''
        first_term  = (x - self.w) ** 2 * np.exp( -self.θ3 * (self.T - t))
        second_term = self.θ2 * t ** 2
        third_term  = self.θ1 * t
        fourth_term = self.θ0
        V = first_term + second_term + third_term + fourth_term
        return V

    def __diff_V_i_(self, D, i):
        ''' 
            This function calculates the approximate derivative of the value function given two 
            consecutives times.

            D is the kth column of the matrix D 
        '''
        t_i         = D[i][0]
        x_i         = D[i][1]
        t_i_plus    = D[i+1][0]
        x_i_plus    = D[i+1][1]
        num_diff    = self.__V_(t_i_plus, x_i_plus) - self.__V_(t_i, x_i)
        diff_V_i    = num_diff / self.dt
        return diff_V_i

    def __H_(self, t):
        return self.ϕ1 + self.ϕ2 * (self.T - t)

    def __gradC_θ1_(self, D):
        sum = 0
        for i in range(len(D)-1):
            diff_V_i  = self.__diff_V_i_(D, i)
            t_i       = D[i][0]
            sum       += (diff_V_i - self.λ * self.__H_(t_i)) * self.dt
        return sum
    
    def __gradC_θ2_(self, D):
        sum = 0
        for i in range(len(D)-1):
            diff_V_i    = self.__diff_V_i_(D, i)
            t_i_plus    = D[i+1][0]
            t_i         = D[i][0]
            sum         += (diff_V_i - self.λ * self.__H_(t_i)) * (t_i_plus ** 2 - t_i ** 2)
        return sum
    
    def __gradC_ϕ1_(self, D):
        sum = 0
        for i in range(len(D)-1):
            diff_V_i  = self.__diff_V_i_(D, i)
            t_i       = D[i][0]
            sum       += (diff_V_i - self.λ * self.__H_(t_i)) * self.dt
        return - self.λ * sum

    def __gradC_ϕ2_(self, D):
        sum = 0
        for i in range(len(D)-1):
            diff_V_i        = self.__diff_V_i_(D, i)
            t_i             = D[i][0]
            x_i             = D[i][0]
            t_i_plus        = D[i+1][0]
            x_i_plus        = D[i+1][0]
            first_factor    = (diff_V_i - self.λ * self.__H_(t_i)) * self.dt
            first_num_2nd_factor  = (x_i_plus - self.w) ** 2 * np.exp(-2 * self.ϕ1 * (self.T - t_i_plus)) * (self.T - t_i_plus)
            second_num_2nd_factor = (x_i - self.w) ** 2 * np.exp(-2 * self.ϕ1 * (self.T - t_i)) * (self.T - t_i)
            num_2nd_factor  = - 2 * (first_num_2nd_factor - second_num_2nd_factor)
            second_factor   = num_2nd_factor / self.dt - self.λ * (self.T - t_i)
            sum += first_factor * second_factor
        return sum

    def __next_wealth_(self, prev_x, prev_u):
        '''
            This function calculates the (total) wealth of an investor from the previous wealth
            and the current amount invested in the risky asset which has been determined following
            the EMV-algorithm.
        '''
        W = np.random.normal(0, 1)
        return prev_x + self.σ * prev_u * (self.ρ * self.dt + W * np.sqrt(self.dt))

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
        for i in range(1, self.final_step):
            # Mean and variance
            pi_mean     = self.__pi_mean_(x)
            pi_variance = self.__pi_variance_(t)
            # u_i
            u  = np.random.normal(pi_mean, pi_variance)
            # t_i
            t  = i * self.dt
            # x_{t_i}
            x  = self.__next_wealth_(x, u)
            # Collected samples
            D.append([t,x])
        return D

    def __update_θ0_(self):
        return - self.θ2 * self.T ** 2 - self.θ1 * self.T - (self.w - self.z) ** 2

    def __update_θ1_(self, D):
        return self.old_θ1 - self.ηθ * self.__gradC_θ1_(D)

    def __update_θ2_(self, D):
        return self.old_θ2 - self.ηθ* self.__gradC_θ2_(D)
    
    def __update_θ3_(self):
        return 2 * self.ϕ2

    def __update_ϕ1_(self, D):
        return self.old_ϕ1 - self.ηϕ * self.__gradC_ϕ1_( D)
    
    def __update_ϕ2_(self,  D):
        return self.old_ϕ2 - self.ηϕ * self.__gradC_ϕ2_(D)

    def __update_SDA_(self, D):
        # θ1, θ2
        self.θ1 = self.__update_θ1_(D)
        self.θ2 = self.__update_θ2_(D)
        # ϕ1, ϕ2
        self.ϕ1 = self.__update_ϕ1_(D)
        self.ϕ2 = self.__update_ϕ2_(D)
        # Related with other parameters
        self.θ0 = self.__update_θ0_()
        self.θ3 = self.__update_θ3_()
        
        # old = new
        self.old_θ0 = self.θ0
        self.old_θ1 = self.θ1
        self.old_θ2 = self.θ2
        self.old_θ3 = self.θ3
        self.old_ϕ1 = self.ϕ1
        self.old_ϕ2 = self.ϕ2
        return self.θ, self.ϕ

    def __update_w_(self, k, final_wealths):
        if k % self.N == 0:
            mean_x = 0
            for j in range(k - self.N + 1, k):
                mean_x += final_wealths[j]
            mean_x /= self.N 
            self.w   -= self.α * ( mean_x - self.z )
        return self.w

    def EMV(self):
        theta = [1,1,1,1]
        phi   = [1,1]
        w     = 1
        # Vector of final wealths (states)
        final_wealths = []
        # Number of iterations
        for k in range(1, self.M):
            # Collected samples (each try we sample a new )
            D           = self.__collect_samples_()
            # Save final-wealth
            final_wealths.append(D[-1][1])
            # Descent-Gradient
            self.θ, self.ϕ  = self.__update_SDA_(D)
            print(k, self.ρ**2, self.θ3)
            # Update w
            self.w           = self.__update_w_(k, final_wealths)
        return self.θ, self.ϕ, self.w