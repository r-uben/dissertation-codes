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

    r = 0.02
    μ = 0.30
    σ  = 0.10
    ρ = (μ - r) / σ

    x_0 = 1
    z = 1.4

    M = 20000
    N = 50
    λ = 2
    ηθ = 0.0005
    ηφ = ηθ
    α  = 0.05

    T = 1
    dt = 1/252

    data = EMV(α, ηθ, ηφ, x_0, z, T, dt, λ, M, N, ρ, σ)
    θ, ϕ, w = data.EMV()