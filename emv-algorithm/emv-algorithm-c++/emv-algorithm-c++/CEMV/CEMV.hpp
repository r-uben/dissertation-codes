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

class CEMV
{
public:
    CEMV(double α, double ηθ, double ηφ, double x0, double z, double T, double dt, double λ, double M, double N, double ρ, double σ);
    void EMV(vector<double>&θ, vector<double>&φ, double w);
    
private:
    // SOME USEFUL FUNCTIONS
    vector <vector<double>> collectSamples(vector<double>&φ, double w);
    double piMean(vector<double>&φ, double x, double w);
    double piVariance(vector<double>&φ, double t);
    double nextWealth(double x, double u);
    double updateLagrange(double w, int k, vector<double> &finalWealths);
    
    // VARIABLES AND PARAMETERS
    double m_α;
    double m_ηθ;
    double m_ηφ;
    double m_x0;
    double m_z;
    double m_T;
    double m_dt;
    double m_λ;
    int m_M;
    int m_N;
    double m_ρ;
    double m_σ;
    double m_finalStep;
};


#endif /* CEMV_hpp */
