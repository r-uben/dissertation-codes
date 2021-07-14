import numpy as np
import pandas as pd
from matplotlib.pylab import plt
from emv import EMV
from parameters import Parameters


def write_vector(file, vector, name):
    file.write(name + " = [")
    for i in range(len(vector)-1):
        file.write(str(vector[i]) + ", ")
    file.write(str(vector[-1]) + "]\n")
    return file

def write_rl_parameters(file, theta, phi, w):
    file = write_vector(file, theta, "θ")
    file = write_vector(file, phi, "φ")
    file.write("w = " +str(w))
    return file

def rel_error_sharpe_ratio(rho, theta3):
    return abs(theta3 - rho ** 2) / (rho ** 2)

if __name__== "__main__":

    μ = [(2*n-1)/10 for n in range(-2,4)]
    σ = [n/10 for n in range(1,5)]

    r = 0.02

    mu = 0.1
    sigma  = 0.1
    ρ = (mu - r) / sigma

    x_0 = 1
    z = 1.4

    M = 200
    N = 10
    λ = 2
    ηθ = 0.0005
    ηφ = ηθ
    α  = 0.05

    T = 1 
    dt = 1/252

    rho2        = [] 
    num_samples = []
    rel_errors_rho  = []

    df = pd.read_csv("/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data/wealth_process.csv")
    print(df)
    t = df["t"]
    x = df["x"]
    plt.plot(t, x)
    plt.show()

    # data = EMV(α, ηθ, ηφ, x_0, z, T, dt, λ, M, N, ρ, sigma)
    # theta, phi, w = data.EMV()
    # print(theta, phi, w)

    #for M in range(10,2000,10):
    #    data = EMV(α, ηθ, ηφ, x_0, z, T, dt, λ, M, N, ρ, sigma)
    #    theta, phi, w = data.EMV()
    #    rel_error_rho   = rel_error_sharpe_ratio(ρ, theta[3])
    #    print(M, ρ, np.sqrt(theta[3]))
    #rel_errors_rho.append(rel_error_rho)
    #num_samples.append(M)
    #plt.plot(num_samples,rel_errors_rho, marker='x', color='red')
