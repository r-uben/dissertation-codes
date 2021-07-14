//
//  CEMV.cpp
//  emv-algorithm-c++
//
//  Created by Rubén Fernández Fuertes on 13/7/21.
//

#include "CEMV.hpp"
#include "CDescendentGradient.hpp"

#include <random>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

EMV::CEMV(double α, double ηθ, double ηφ, double x0, double z, double T, double dt, double λ, double M, double N, double ρ, double σ)
{
    m_α     = α;
    m_ηθ    = ηθ;
    m_ηφ    = ηφ;
    m_x0    = x0;
    m_z     = z;
    m_T     = T;
    m_dt    = dt;
    m_λ     = λ;
    m_M     = M;
    m_N     = N;
    m_ρ     = ρ;
    m_σ     = σ;
    m_finalStep = int(floor(m_T / m_dt));
}

// Main algorithm for learning the parameters
void
EMV::emv(vector<double>&init_θ, vector<double>&init_φ, double init_w)
{
    // Initial RL parameters
    m_w = init_w;
    m_θ = init_θ;
    m_φ = init_φ;
    // Write the collected sample in a .csv file
    ofstream output;
    output.open("/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data/wealth_process.csv");
    OUTPUT "k" COMMA "t" COMMA "u" COMMA "x" END_LINE
    // Initial sample
    for (int k = 1; k <= m_M; k++)
    {
        // Collected samples (complete path)
        collectSamples(k, output);
        // Save Final Wealths
        m_finalWealths.push_back(m_D.back()[1]);
        // Update φ, θ, w:
        updateSDEparameters();
        updateLagrange(k);
    }
    output.close();
}

void
EMV::collectSamples(int k, ofstream &output)
{
    double t = 0., x = m_x0, u;
    vector <double> sample = { t, x };
    OUTPUT k COMMA t COMMA "-" COMMA x END_LINE
    m_D = { sample };
    // Mean and Variance for the initial sample (recall that π depends on t, x and w.
    piMean(x);
    piVariance(t);
    // Collect the sample
    for ( int i = 1; i <= m_finalStep; i++ )
    {
        // Distribution
        mt19937 rng;
        normal_distribution<double> normal(m_piMean, m_piVar);
        // Update risky allocation, next time step and wealth
        u = normal(rng);
        t = i * m_dt;
        x = nextWealth(x, u);
        OUTPUT k COMMA t COMMA u COMMA x END_LINE
        sample = { t, x };
        m_D.push_back(sample);
        // Update the Mean and Variance
        piMean(x);
        piVariance(t);
    }
}

void
EMV::piMean(double x)
{
    double φ1, φ2;
    double first_factor, second_factor, coeff;
    // We want to calculate the mean of the distribution πφ
    // Each parameter of the vector of parameters to be learnt, φ
    φ1 = m_φ[0];
    φ2 = m_φ[1];
    // To  clean up the code we just separate the result into different factors
    first_factor    = sqrt( (2. * φ2) / (m_λ * PI) );
    second_factor   = exp( (2. * φ1 - 1.) / 2.);
    coeff           = - first_factor * second_factor;
    // Return the man of the policy (density)
    m_piMean = coeff * (x - m_w);
}

void
EMV::piVariance(double t)
{
    double φ1, φ2;
    double num_term, exp_term;
    // We want to calculate the variance of the distribution πφ
    // Each parameter of the list of parameters phi
    φ1 = m_φ[0];
    φ2 = m_φ[1];
    // To clean up the code we just separate the result into different factors
    num_term = 1. / (2. * PI);
    exp_term = exp(2. * φ2 * (m_T - t) + 2. * φ1 - 1.);
    // Return the variance of the policy (density)
    m_piVar =  num_term * exp_term;
}

double
EMV::nextWealth(double x, double u)
{
    double nextX, ε;
    static mt19937 rng;
    // Normal distribution
    normal_distribution<double> normal(0.,1.);
    ε = normal(rng);
    // NextX
    nextX = x + m_σ * u * (m_ρ * m_dt + ε * sqrt(m_dt));
    return nextX;
}

void
CEMV::updateLagrange(int k)
{
    double mean_x;
    if (k % m_N == 0) {
        mean_x = 0;
        for (int j = k - m_N + 1; j <= m_M; j++) {
            mean_x += m_finalWealths[j];
        }
        mean_x /= m_N;
        m_w -= m_α * ( mean_x - m_z);
    }
}

void
EMV::updateSDEparameters()
{
    // Update φ, θ:
    SDA parameters(m_ηθ, m_ηφ, m_z, m_λ, m_θ, m_φ, m_w, m_D);
    parameters.updateAll();
    m_θ = parameters.Getθ();
    m_φ = parameters.Getφ();
}
