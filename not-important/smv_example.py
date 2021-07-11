# Here we present a brief analysis of a geometric analysis of E-S0 efficient porotfolios based on the historic
# of the three securities in Table 1, p. 195

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/table1-p.195.csv", sep=';')

# The standard conditions here are just x_1 + x_2 + x_3 = 1, x_j >= 0 for all j = 1,2,3,
# where x_j indicates the amount we invest in the jth secutiry.

def semi_variance(vector):
    ''' Function to calculate the semi-variance of a vector '''
    sum = 0
    for v in vector:
        if v <= 0:
            sum += v ** 2
    return sum / len(v)

def semi_covariance(vector1, vector2):

    ### Control condition
    if len(vector1) != len(vector2):
        return 'The two vectors must have the same size'

    ### Calculate semi-variance
    sum = 0
    N = len(vector1)
    for n in range(N):
        v1 = vector1[n]
        v2 = vector2[n]
        if v1*v2 <= 0:
            sum += v1*v2

    return sum / N

def product_vector(vector1, vector2):

    ''' Function to calculate the vector which consists of the product
    of each component of each vector '''
    
    ### Control condition
    if len(vector1) != len(vector2):
        return 'The two vectors must have the same size'

    ### Create the vector
    N = len(vector1)
    new_vector = [vector1[n]*vector2[n] for n in range(N)]

    return new_vector



