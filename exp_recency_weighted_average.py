import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rc

# TEX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

path1 = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-tex/img/'
path2 = '/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-img'

# REWARD WEIGHTS FUNCTION
def reward_weights(n, i, alpha):
    return alpha * (1 - alpha) ** (n - i)

# VALUE WEIGHTS FUNCTION
def value_weights(n, alpha):
    return (1 - alpha) ** n 

# TOTAL WEIGHT (MUST BE EQUAL TO 1)
def total_weights(n, alpha):
    rewards = 0
    for i in range(1, n+1):
        rewards += reward_weights(n, i, alpha)
    return value_weights(n, alpha) + rewards

figure(figsize=(10, 6), dpi=2000)

# PLOT SOME REWARDS FUNCTION FOR DIFFERENT N's
for n in [5,6,7,10]:
    alpha = 0.3
    K = [k for k in range(1,n+1)]
    R = [reward_weights(n, k, alpha) for k in K]
    plt.plot(K, R, label='n='+str(n))
    plt.legend(loc='upper left', fontsize=13)

# LABELS AND OTHER STUFF
plt.ylabel(r'$(1-\alpha)^{n-k}$', labelpad=10, fontsize=15)
plt.xlabel(r'$k$', labelpad=10, fontsize=15)

# AXES
axes = plt.gca()
axes.set_xlim([1,10])
axes.set_ylim([0,0.30])

plt.savefig(path1 + 'exp_recency_weighted_average.eps', format='eps')