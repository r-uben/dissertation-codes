//
//  CEMV.hpp
//  emv-algorithm-c++
//
//  Created by Rubén Fernández Fuertes on 13/7/21.
//

#ifndef CEMV_hpp
#define CEMV_hpp


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#define PI atan(1)*4
using namespace std;

/*
 This is the Reinforcement Learning (RL) algorithm for the Exploratory Mean Variance (EMV) Problem (Wang & Zhou, 2020) in one-dimension (one risky asset).
 
 INPUTS:
 // LEARNING RATES
 -- α:  Learning rate for the TD update for the Lagrange Multiplier, w.
 -- η0: Learning rate for the SDE update for the vector θ.
 -- ηφ: Learning rate for the SDE update for the vector φ.
 
 // PORTFOLIO
 -- x0: Initial wealth.
 -- z:  Target value.
 -- T:  Terminal date.
 -- dt: Time-steps length.
 
 // MORE PARAMETERS FOR THE ALGORITHM
 -- λ:  Temperature parameter (it measures the exploration: importance of the entropy term).
 -- M:  Number of episodes.
 -- N:  Sample size.
 
 // MARKET SIMULATOR "Market"
 -- ρ:  Sharpe Ratio.
 -- σ:  Volatility
 
 OUTPUTS:
 -- θ:  Contains the parameters for the approximation of the (optimal) value function, Vφ.
 -- φ:  Contains the parameters for the approximation of the mean and variance of the policy distribution, πφ.
 -- w:  Lagrange Multiplier.
 */

class CEMV
{
public:
    CEMV(double α, double ηθ, double ηφ, double x0, double z, double T, double dt, double λ, double M, double N, double ρ, double σ);
    
    // EMV ALGORITHM
    void emv(vector<double>&init_θ, vector<double>&init_φ, double init_w);
    
private:
    // SOME USEFUL FUNCTIONS
    // SAMPLES
    void    collectSamples(int k, ofstream &output);
    void    piMean(double x);
    void    piVariance(double t);
    double  nextWealth(double x, double u);
    // RL
    void    updateSDEparameters();
    void    updateLagrange(int k);
    
    // VARIABLES AND PARAMETERS
    // LEARNING RATES
    double m_α;
    double m_ηθ;
    double m_ηφ;
    // PORTFOLIO AND ITS TARGETS
    double m_x0;
    double m_z;
    double m_T;
    double m_dt;
    // TEMPERATURE PARAMETER
    double m_λ;
    // NUMBER OF EPISODES
    int m_M;
    // SAMPLE SIZE
    int m_N;
    // MARKET
    double m_ρ;
    double m_σ;
    double m_finalStep;
    // POLICY
    double m_piMean;
    double m_piVar;
    // COLLECTED SAMPLES
    vector<vector<double>> m_D;
    vector<double> m_finalWealths;
    // RL PARAMETERS
    vector <double> m_θ;
    vector <double> m_φ;
    double m_w;
};

#define COMMA               << "," << 
#define START_LINE          cout <<
#define END_LINE            << endl;
#define OUTPUT              output <<

#define EMV CEMV

#endif /* CEMV_hpp */
