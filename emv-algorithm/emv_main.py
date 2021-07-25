import pandas as pd
from emv import EMV

def read_real_data(name):
    path = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data/real-data/'
    return pd.read_csv(path + name + '.csv', sep=',', index_col=0)

def clean_dataset(dataset):
    for key in dataset.keys():
        dataset[key] = dataset[key].fillna(dataset[key].mean())
    return dataset
if __name__== "__main__":

    r = 0.02
    μ = 0.30
    σ = 0.10
    ρ = (μ - r) / σ

    x_0 = 1
    z = 1.4

    M = 20000
    N = 20
    λ = 10
    ηθ = 0.00005
    ηφ = ηθ
    α  = 0.05


    market = 'BTC-USD_25:07:2020_25:07:2021' 
    dataset = clean_dataset(read_real_data(market))
    T = 1
    dt = 1/len(dataset['Close'])
    data = EMV(market, α, ηθ, ηφ, x_0, z, T, dt, λ, M, N, μ, σ, r, dataset)
    data.EMV()