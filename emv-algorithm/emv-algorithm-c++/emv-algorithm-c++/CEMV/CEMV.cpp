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

CEMV::CEMV(double α, double ηθ, double ηφ, double x0, double z, double T, double dt, double λ, double M, double N, double ρ, double σ)
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
CEMV::EMV(vector<double>&θ, vector<double>&φ, double w)
{
    vector <double> finalWealths;
    vector <vector<double>> D;
    for (int k = 1; k <= m_M; k++)
    {
        // Collected samples (complete path)
        D = collectSamples(φ, w);
        // Save Final Wealths
        finalWealths.push_back(D.back()[1]);
        // Update φ, θ:
        SDA update(m_ηθ, m_ηφ, m_z, m_λ, θ, φ, w, D);
        θ = update.Getθ();
        φ = update.Getφ();
        w = updateLagrange(w, k, finalWealths);
    }
    cout << m_ρ * m_ρ << θ[3] << endl;
}

vector<vector<double>>
CEMV::collectSamples(vector<double>&φ, double w)
{
    // Initial sample
    double t = 0., x = m_x0, u;
    vector <double> sample = { t, x };
    vector <vector <double>> D = { sample };
    for ( int i = 1; i <= m_finalStep; i++)
    {
        // Mean and Variance
        double mean = piMean(φ, x, w);
        double var  = piVariance(φ, t);
        // Distribution
        static mt19937 rng;
        normal_distribution<double> phi(mean,var);
        // Update risky allocation, next time step and wealth
        u = phi(rng);
        t = i * m_dt;
        x = nextWealth(x, u);
        sample = { t, x };
        D.push_back(sample);
    }
    return D;
}

double
CEMV::piMean(vector<double>&φ, double x, double w)
{
    // We want to calculate the mean of the distribution πφ
    double piMean;
    // Each parameter of the vector of parameters to be learnt, φ
    double φ1 = φ[0];
    double φ2 = φ[1];
    // To  clean up the code we just separate the result into different factors
    double first_factor    = sqrt( (2. * φ2) / (m_λ * PI));
    double second_factor   = exp( (2. * φ1 - 1.) / 2.);
    double coeff           = - first_factor * second_factor;
    // Return the man of the policy (density)
    piMean = coeff * (x - w);
    return piMean;
}

double
CEMV::piVariance(vector<double>&φ, double t)
{
    // We want to calculate the variance of the distribution πφ
    double piVar;
    // Each parameter of the list of parameters phi
    double φ1 = φ[0];
    double φ2 = φ[1];
    // To clean up the code we just separate the result into different factors
    double num_term = 1. / (2. * PI);
    double exp_term = exp(2. * φ2 * (m_T - t) + 2. * φ1 - 1.);
    // Return the variance of the policy (density)
    piVar =  num_term * exp_term;
    return piVar;
}

double
CEMV::nextWealth(double x, double u)
{
    double nextX;
    static mt19937 rng;
    // Normal distribution
    normal_distribution<double> phi(0.,1.);
    // NextX
    nextX = x + m_σ * u * (m_ρ * m_dt + phi(rng) * sqrt(m_dt));
    return nextX;
}

double 
CEMV::updateLagrange(double w, int k, vector<double> &finalWealths)
{
    if (k % m_N == 0) {
        double mean_x = 0;
        for (int j = k - m_N + 1; j <= m_M; j++) {
            mean_x += finalWealths[j];
        }
        mean_x /= m_N;
        w -= m_α * ( mean_x - m_z);
    }
    return w;
}
