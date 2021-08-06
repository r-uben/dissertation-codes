import pandas as pd
import numpy as np
from d_emv import EMV

def read_real_data(name):
    path = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data/real-data/'
    return pd.read_csv(path + name + '.csv', sep=',', index_col=0)

def clean_dataset(dataset):
    for key in dataset.keys():
        dataset[key] = dataset[key].fillna(dataset[key].mean())
    return dataset
if __name__== "__main__":
    
    σ = np.array([[0.3, 0.23, 0.1], [0.15, 0.2, 0.04], [0.12, 0.8, 0.2]])
    ρ = np.array([0.3, 0.5, 0.1])
    d = 3

    ηw = 0.05
    ηα = 0.0005 
    ηβ = ηα
    ηθ = ηβ
    
    x_0 = 1
    z   = 1.4
    T   = 1
    dt  = 1/252

    λ   = 0.7

    M   = 100
    N   = 20


    market = 'BTC-USD_25:07:2020_25:07:2021' 
    dataset = clean_dataset(read_real_data(market))
    T = 1
    dt = 1/252#len(dataset['Close'])
    data = EMV(d, 'log', ηw, ηα, ηβ, ηθ, x_0, z, T, dt, λ, M, N, ρ, σ)
    data.EMV()