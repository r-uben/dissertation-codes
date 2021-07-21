import numpy as np
import pandas as pd
from matplotlib.pylab import plt
from emv import EMV
from parameters import Parameters


if __name__== "__main__":

    r = 0.02
    μ = 0.50
    σ = 0.20
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