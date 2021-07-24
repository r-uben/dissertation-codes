import numpy as np
import pandas as pd

from matplotlib.pylab import plt
from emv import EMV
from parameters import Parameters

def read_real_data(name):
    path = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data/real-data/'
    return pd.read_csv(path + name + '.csv', sep=',')

if __name__== "__main__":

    r = 0.02
    μ = 0.30
    σ = 0.10
    ρ = (μ - r) / σ

    x_0 = 1
    z = 1.4

    M = 20000
    N = 20
    λ = 2
    ηθ = 0.0005
    ηφ = ηθ
    α  = 0.05

    T = 1
    dt = 1/252

    market = 'SAN_24:07_year' 
    df = read_real_data(market)
    data = EMV(market, α, ηθ, ηφ, x_0, z, T, dt, λ, M, N, μ, σ, r, df)
    data.EMV()