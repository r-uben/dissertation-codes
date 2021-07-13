import numpy as np
from scipy.stats import norm

class EMV(object):

    def __init__(self, alpha, eta_theta, eta_phi, x_0, z, T, dt, λ, M, N, sharpe_ratio, sigma):
        # Learning rates
        self.alpha          = alpha
        self.eta_theta      = eta_theta
        self.eta_phi        = eta_phi
        # Initial wealth
        self.x_0            = x_0
        # Target Payoff
        self.z              = z
        # Investment Horizon
        self.T              = T
        # Discretisation
        self.dt             = dt
        # Exploration Rate
        self.λ              = λ
        # Number of Iterations
        self.M              = M
        # Sample Average Size
        self.N              = N
        # Final Step
        self.final_step = int(np.floor(self.T / self.dt))
        # MARKET (PARAMETERS)
        self.sharpe_ratio   = sharpe_ratio
        self.sigma          = sigma

    def __take_row_(self, D, k):
        D_k = []
        for key in D:
            if key == k:
                D_k.append(D[key])
            else:
                print('There is not such a key')
        return D_k

    def __pi_mean_(self, phi, x, w):
        # Each parameter of the list of parameters phi
        phi1 = phi[0]
        phi2 = phi[1]
        # To clean up the code we just separate the result into different factors
        first_factor    = np.sqrt( (2 * phi2) / (self.λ * np.pi))
        second_factor   = np.exp( (2 * phi1 - 1) / 2) 
        coeff           = - first_factor * second_factor
        # Return the mean of the policy (density)
        mean = coeff * (x - w)
        return mean

    def __pi_variance_(self, phi, t):
         # Each parameter of the list of parameters phi
        phi1 = phi[0]
        phi2 = phi[1]
         # To clean up the code we just separate the result into different factors
        num_term = 1 / (2 * np.pi)
        exp_term = np.exp(2 * phi2 * (self.T - t) + 2 * phi1 - 1)
        # Return the variance of the policy (density)
        variance =  num_term * exp_term
        return variance

    def __V_(self, theta, t, x, w):
        ''' 
            This function calculates the value function for a certain wealth in a certain moment    
        '''
        first_term  = (x - w) ** 2 * np.exp(-theta[3] * (self.T - t))
        second_term = theta[2] * t ** 2
        third_term  = theta[1] * t
        fourth_term = theta[0]
        V = first_term + second_term + third_term + fourth_term
        return V

    def __diff_V_i_(self, theta, w, D, i):
        ''' 
            This function calculates the approximate derivative of the value function given two 
            consecutives times.

            D is the kth column of the matrix D 
        '''
        t_i         = D[i][0]
        x_i         = D[i][1]
        t_i_plus    = D[i+1][0]
        x_i_plus    = D[i+1][1]
        num_diff    = self.__V_(theta, t_i_plus, x_i_plus, w) - self.__V_(theta, t_i, x_i, w)
        diff_V_i    = num_diff / self.dt
        return diff_V_i

    def __C_diff_theta1_(self, phi, theta, w, D):
        sum = 0
        for i in range(len(D_k)-1):
            diff_V_i    = self.__diff_V_i_(theta, w, D, i)
            t_i         = D[i][0]
            phi1        = phi[0]
            phi2        = phi[1]
            sum         += (diff_V_i - self.λ * (phi1 + phi2 * (self.T - t_i))) * self.dt
        return sum
    
    def __C_diff_theta2_(self, phi, theta, w, D):
        sum = 0
        for i in range(len(D)-1):
            diff_V_i    = self.__diff_V_i_(theta, w, D, i)
            t_i_plus    = D[i+1][0]
            t_i         = D[i][0]
            sum         += (diff_V_i - self.λ * (phi[0] + phi[1] * (self.T - t_i))) * (t_i_plus ** 2 - t_i ** 2)
        return sum
    
    def __C_diff_phi_1_(self, phi, theta, w, D):
        return - self.λ * self.__C_diff_theta1_(phi, theta, w, D_k)

    def __C_diff_phi_2_(self, phi, theta, w, D):
        sum = 0
        for i in range(len(D)-1):
            diff_V_i        = self.__diff_V_i_(theta, w, D, i)
            t_i             = D[i][0]
            x_i             = D[i][0]
            t_i_plus        = D[i+1][0]
            x_i_plus        = D[i+1][0]
            first_factor    = (diff_V_i - self.λ * (phi[0] + phi[1] * (self.T - t_i))) * self.dt
            first_num_2nd_factor  = (x_i_plus - w) ** 2 * np.exp(-2 * phi[1] * (self.T - t_i_plus)) * (self.T - t_i_plus)
            second_num_2nd_factor = (x_i - w) ** 2 * np.exp(-2 * phi[1] * (self.T - t_i)) * (self.T - t_i)
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
        return prev_x + self.sigma * prev_u * (self.sharpe_ratio * self.dt + W * np.sqrt(self.dt))

    def __collect_samples_(self, phi, w):
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
            pi_mean     = self.__pi_mean_(phi, x, w)
            pi_variance = self.__pi_variance_(phi, t)
            # u_i
            u  = np.random.normal(pi_mean, pi_variance)
            # t_i
            t  = i * self.dt
            # x_{t_i}
            x  = self.__next_wealth_(x, u)
            # Collected samples
            D.append([t,x])
        return D

    def __uptate_theta0_(self, theta, w):
        theta1 = theta[1]
        theta2 = theta[2]
        return - theta2 * self.T ** 2 - theta1 * self.T - (w - self.z) ** 2

    def __update_theta1_(self, phi, theta, w, D_k):
        return theta[1] - self.eta_theta * self.__C_diff_theta1_(phi, theta, w, D_k)

    def __update_theta2_(self, phi, theta, w, D_k):
        return theta[2] - self.eta_theta * self.__C_diff_theta2_(phi, theta, w, D_k)
    
    def __update_theta3_(self, phi):
        phi2 = phi[1]
        return 2 * phi2

    def __update_phi1_(self, phi, theta, w, D_k):
        return phi[0] - self.eta_phi * self.__C_diff_phi_1_(phi, theta, w, D_k)
    
    def __update_phi2_(self, phi, theta, w, D_k):
        return phi[1] - self.eta_phi * self.__C_diff_phi_2_(phi, theta, w, D_k)

    def __update_all_(self, phi, theta, w, D):
        for i in range(1, self.final_step):
            # θ1, θ2
            theta[1] = self.__update_theta1_(phi, theta, w, D)
            theta[2] = self.__update_theta2_(phi, theta, w, D)
            # Related with other parameters
            theta[0] = self.__uptate_theta0_(theta, w)
            theta[3] = self.__update_theta3_(phi)
            # ϕ1, ϕ2
            phi[0]   = self.__update_phi1_(phi, theta, w, D)
            phi[1]   = self.__update_phi2_(phi, theta, w, D)
        return phi, theta

    def __update_w_(self, w, k, final_wealths):
        if k % self.N == 0:
            mean_x = 0
            for j in range(k - self.N + 1, k):
                mean_x += final_wealths[j]
            mean_x /= self.N 
            w   -= self.alpha * ( mean_x - self.z )
        return w

    def EMV(self):
        theta = [2,2,2,3]
        phi   = [1,1]
        w     = 1
        # Vector of final wealths (states)
        final_wealths = []
        # Number of iterations
        for k in range(1, self.M):
            # Collected samples (each try we sample a new )
            D           = self.__collect_samples_(phi, w)
            # Save final-wealth
            final_wealths.append(D[-1][1])
            # Descent-Gradient
            phi, theta  = self.__update_all_(phi, theta, w, D)
            print(self.sharpe_ratio, self.sharpe_ratio**2, np.sqrt(theta[3]))
            # Update w
            w           = self.__update_w_(w, k, final_wealths)
        return theta, phi, w