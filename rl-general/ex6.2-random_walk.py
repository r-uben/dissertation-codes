import numpy as np
import matplotlib.pyplot as plt
import random

def get_item(desired_value, dictionary):
    for item, value in dictionary.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value == desired_value:
            return item

def coord_alphabet():
    alphabet = {"A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
    "E" : 4,
    "F" : 5,
    "G" : 6,
    "H" : 7, 
    "I" : 8,
    "J" : 9,
    "K" : 10,
    "L" : 11,
    "M" : 12,
    "N" : 13,
    "O" : 14,
    "P" : 15,
    "Q" : 16,
    "R" : 17,
    "S" : 18,
    "T" : 19,
    "U" : 20,
    "V" : 21,
    "W" : 22,
    "X" : 23,
    "Y" : 24,
    "Z" : 25
    }
    return alphabet

def rewards():
    rewards = {"left": 0, 
    "A" : 0,
    "B" : 0,
    "C" : 0,
    "D" : 0,
    "E" : 0,
    "right" : 1}
    return rewards

def π():
    action = random.randint(0, 1)
    if action == 0:
        action -= 1
    return action

def coordinates(state, rewards):
    alphabet        = coord_alphabet()
    new_alphabet    = {}
    last_letter     = 0
    for letter in alphabet:
        if letter in rewards:
            new_alphabet[letter] = alphabet[letter] - alphabet[state]
            last_letter          += 1
    # Append extremes
    last_letter = get_item(last_letter-1, alphabet)
    new_alphabet["left"] = new_alphabet["A"] - 1
    new_alphabet["right"] = new_alphabet[last_letter] + 1
    return new_alphabet

def homogeneous_initial_value(value, coord):
    '''
        Arbitrary initial values except V(terminal) = 0
    '''
    values = {}
    for state in coord:
        if state == 'left' or state == 'right':
            values[state] = 0
        else:
            values[state] = value
    return values 
    
def episode(S, α, γ, rewards, values):
    # Center the origin in the initial state
    coord   = coordinates(S, rewards)
    walk    = []
    while True:
        # Take (random) action (left or right)
        A = π()
        # Keep a record of the walk
        walk.append(A)
        # Update step
        new_S = get_item(sum(walk), coord)
        # V <- V + α (R + γ V(S') - V(S))
        values[S] += α * (rewards[new_S] + γ * values[new_S] - values[S])
        # S <- S'
        S = new_S
        # If S is terminal, stop
        if S == "left" or S == "right":
            break
    return values

def E(vector):
    return sum(vector) / len(vector)

def RMS(theor, emp):
    sq = []
    for v in emp:
        sq.append((theor - v) ** 2)
    return np.sqrt(E(sq))

if __name__== "__main__":
    print("-------------------------------------")
    print("-------------------------------------")
    # Dictionary of rewards
    R = rewards()
    # Learning rate 
    α = 0.01
    # Discount factor
    γ = 1  
    # Number of episodes
    M = 100
    # THEOR VALUES (CALCULATED AD HOC)
    theor_values = [1/6, 2/6, 3/6, 4/6, 5/6]
    mean_theor_value = E(theor_values)
    # Different Learning rates
    for α in [0.01, 0.03, 0.4, 0.5, 0.05, 0.1, 0.15]:
        # Number of episodes
        episodes    = []
        rms         = []
        for M in range(1, 100):
            # 100 times with M episodes to calculate RMS
            mean_values = []
            for _ in range(200):
                # Dictionary of initial values
                V  = homogeneous_initial_value(0.5, R)
                # Loop for each episode (M times)
                for _ in range(M):
                    # Initial state, S
                    S = "C"
                    # Run episode
                    V = episode(S, α, γ, R, V)
                # Keep all values in all state
                values = [V['A'], V['B'], V['C'], V['D'], V['E']]
                # Calculate its mean
                mean_value = E(values)
                # Keep the mean
                mean_values.append(mean_value)
            # Calculate and keep rms values 
            rms.append(RMS(mean_theor_value, mean_values))
            # Keep number of episodes
            episodes.append(M)
        # Plot
        plt.plot(episodes, rms, label = r'$α=$' + str(α))
    # Legend
    plt.legend()
    # Show plot
    plt.show()


            
        
    
    

