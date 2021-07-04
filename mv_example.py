import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rc

def _get_index_max__(vector):
    ''' Takes the index of the maximum entry of a vector '''
    v       = vector[0]
    index   = 0
    for i in range(len(vector)):
        if vector[i] >= v:
            v     = vector[i]
            index = i
    return index

def _newX__(index, n):
    ''' Creates a vector with a 1 in index and 0 everywhere else '''
    L = []
    for j in range(n):
        if j == index:
            L.append([1])
        if j != index:
            L.append([0])
    return np.matrix(L)

def _tildeM__(M, indices):
    ''' Updates M with j's which are out '''
    dim = M.shape
    I = dim[0]
    J = dim[-1]
    for k in indices:
        # Rows
        for i in range(I):
            M[i,k-1] = 0
        # Columns
        for j in range(J):
            M[k-1,j] = 0
        # Diagonal
        M[k-1,k-1] = 1
    return M

def _zeros_Out__(M, indices):
    for j in indices:
        M[j-1,j-1] = 0
    return M

def _critical_line__(m, n, x):
    return m * x + n


n = 3

### COVARIANCE MATRIX
C1 = [.0146, .0187, .0145]
C2 = [.0187, .0854, .0104]
C3 = [.0145, .0104, .0289]

C   = np.matrix([C1, C2, C3])

### EXPECTED RETURNS
mu1 = [.062]
mu2 = [.146]
mu3 = [.128]

mu  = np.matrix([mu1, mu2, mu3])

### SYSTEM
A   = np.matrix([1, 1, 1])

M   = np.concatenate((C, np.transpose(A)), axis=1)
aux = np.concatenate((A, np.matrix([0])), axis=1)
M   = np.concatenate((M, aux), axis=0)
b   = np.matrix([[1]])
R   = np.matrix([[0], [0], [0], [1]])
S   = np.concatenate((mu, np.matrix([0])))

### STEP 1: FIND THE EFFICIENT PORTFOLIO X WITH MAXIMUM E
# The portfolio with maximum E consists entirely of the securrity with the
# maximum μ.

max_mu  = max(mu)
j_      = _get_index_max__(mu)
newX    = _newX__(j_,n)

# When more general set of equations AX = b constraints the choice of portfolio,
# it may be necessary to use *linear programming* to find the portfolio which
# maximises E.


### STEP 2: FIND THE FORMULA FOR THE CRITICAL LINE ASSOCIATED WITH newX

indices     = [1,3]
new_M       = _tildeM__(M, indices)
inv_new_M   = np.linalg.inv(new_M)
N           = _zeros_Out__(inv_new_M, indices)

n           = N * R
m           = N * S

print(_critical_line__(m, n, 0.3))

lambdas = np.linspace(-10,10)

#### STEP 3: FIND THE VALUES OF λ_E at which our first critical line
#### intersects each critical line with the following three properties
#### a) all variables are *in* which were *in* on the first critical line
#### b) one additional variable is also *in* 
#### c) all other variables are out

