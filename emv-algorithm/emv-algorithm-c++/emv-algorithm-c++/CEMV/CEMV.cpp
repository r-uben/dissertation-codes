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
    // LEARNING RATES
    m_α     = α;
    m_ηθ    = ηθ;
    m_ηφ    = ηφ;
    // PORTFOLIO AND ITS TARGETS
    m_x0    = x0;
    m_z     = z;
    m_T     = T;
    m_dt    = dt;
    m_finalStep = int(floor(m_T / m_dt));
    // TEMPERATURE PARAMETER
    m_λ     = λ;
    // NUMBER OF EPISODES
    m_M     = M;
    // SAMPLE SIZE
    m_N     = N;
    // MARKET
    m_ρ     = ρ;
    m_σ     = σ;
}

// Main algorithm for learning the parameters
void
EMV::emv(vector<double>&init_θ, vector<double>&init_φ, double init_w)
{
    // Initial RL parameters
    m_w = init_w;
    // θ:
    m_θ = init_θ;
    m_θ0 = m_θ[0];
    m_θ1 = m_θ[1];
    m_θ2 = m_θ[2];
    m_θ3 = m_θ[3];
    // φ:
    m_φ = init_φ;
    m_φ1 = m_φ[0];
    m_φ2 = m_φ[1];
    // m_finalWealths must be an empty vector
    m_finalWealths.clear();
    // Write the collected sample in a .csv file
    ofstream output, output2;
    //    // Wealth processes
    //    output.open("/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data/wealth_process.csv");
    //    // Approximation of ρ^2
    //    output2.open("/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/dissertation/dissertation-codes/data/squared_rho_approximations.csv");
    //    // Title
    //    OUTPUT  "k" COMMA "t"   COMMA "u"  COMMA "x"     END_LINE
    //    OUTPUT2 "k" COMMA "ρ^2" COMMA "θ3" COMMA "Error" END_LINE
    // Run
    for (int k = 1; k <= m_M; k++)
    {
        KTH_ITERATION(k)
        // Calculate the Error between ρ^2 and θ3
        θ3Error();
        //        OUTPUT2    k                     COMMA pow(m_ρ,2)       COMMA m_θ3                      COMMA m_θ3error END_LINE
        START_LINE TAG("ρ^2") pow(m_ρ,2) COMMA TAG(" θ3") m_θ3  COMMA TAG(" Error") m_θ3error   END_LINE
        START_LINE TAG("φ1") m_φ1      COMMA TAG(" φ2") m_φ1  END_LINE
        // Collected samples (complete path)
        collectSamples(k, output);
        // Update φ, θ, w:
        updateSDEparameters();
        // Update Lagrange
        updateLagrange(k);
    }
    //    output.close();
    //    output2.close();
    START_LINE " FINAL ANSWER " END_LINE
    START_LINE TAG("k") m_M COMMA TAG(" ρ^2") pow(m_ρ,2) COMMA TAG(" θ3") m_θ[3] END_LINE
}

void
EMV::collectSamples(int k, ofstream &output)
{
    double t = 0., x = m_x0, u;
    //    OUTPUT k COMMA t COMMA "-" COMMA x END_LINE
    vector <double> sample = { t, x };
    m_D.clear();
    // Collect the sample
    for ( int i = 1; i <= m_finalStep; i++ )
    {
        // Mean and Variance
        piMean(x);
        piVariance(t);
        // Distribution
        normal_distribution<double> Ν(m_piMean, sqrt(m_piVar));
        // START_LINE m_piMean COMMA m_piVar END_LINE
        // Update risky allocation, next time step and wealth
        u = Ν(m_random);
        x = nextWealth(x, u);
        t = i * m_dt;
        // OUTPUT k COMMA t COMMA u COMMA x END_LINE
        sample = { t, x };
        m_D.push_back(sample);
    }
    // Save Final Wealths
    m_finalWealths.push_back(x);
}

// Mean of the distribution πφ
void
EMV::piMean(double x)
{
    double first_factor, second_factor, coeff;
    first_factor    = sqrt( (2. * m_φ2) / (m_λ * PI) );
    second_factor   = exp( m_φ1 - 0.5 );
    coeff           = - first_factor * second_factor;
    // Return the man of the policy (density)
    m_piMean = coeff * (x - m_w);
}

// Variance of the distribution πφ
void
EMV::piVariance(double t)
{
    double num_term, exp_term;
    num_term = 1. / (2. * PI);
    exp_term = exp( 2. * m_φ2 * (m_T - t) + 2. * m_φ1 - 1. );
    m_piVar =  num_term * exp_term;
}

// Given policy, u, and wealth x, provide the next wealth
double
EMV::nextWealth(double x, double u)
{
    double nextX, dW;
    // Normal distribution
    normal_distribution<double> normal(0.,1.);
    dW =  sqrt(m_dt) * normal(m_random);
    // NextX
    nextX = x + m_σ * u * (m_ρ * m_dt + dW);
    return nextX;
}

// Update the Lagrange multiplier
void
EMV::updateLagrange(int k)
{
    double mean_x;
    if (k % m_N == 0) {
        mean_x = 0.;
        for (int j = k - m_N + 1; j <= k; j++)
            mean_x += m_finalWealths[j];
        mean_x /= m_N;
        m_w -= m_α * ( mean_x - m_z );
    }
}

void
EMV::updateSDEparameters()
{
    // Update φ, θ:
    SDA parameters(m_ηθ, m_ηφ, m_z, m_λ, m_θ, m_φ, m_w, m_D);
    parameters.updateAll();
    // Update θ:
    m_θ = parameters.Getθ();
    m_θ0 = m_θ[0];
    m_θ1 = m_θ[1];
    m_θ2 = m_θ[2];
    m_θ3 = m_θ[3];
    // Update φ:
    m_φ = parameters.Getφ();
    m_φ1 = m_φ[0];
    m_φ2 = m_φ[1];
}

void
EMV::θ3Error()
{
    m_θ3error = abs(m_θ[3] - m_ρ * m_ρ) / m_ρ / m_ρ;
}
