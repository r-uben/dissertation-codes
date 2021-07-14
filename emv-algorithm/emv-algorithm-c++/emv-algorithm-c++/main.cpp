//
//  main.cpp
//  emv-algorithm-c++
//
//  Created by Rubén Fernández Fuertes on 13/7/21.
//

// To calculate the time
#include <chrono>
#define  CHRONO   std::chrono
#define  SET_TIME CHRONO::system_clock::now()
#define  DURATION CHRONO::duration
#define  MILLI    std::milli

#include <iostream>
#include <cmath>

#include "CEMV.hpp"

using namespace std;

int main() {
    double α = 0.05, ηθ = 0.0005, ηφ = 0.0005;
    double x0 = 1, z = 1.4, T = 1., dt = 1./252*T;
    double λ = 2;
    int M = 5, N = 10;
    double μ = 0.5, σ = 0.1;
    double r = 0.02;
    double ρ = (μ - r) / σ;
    
    vector <double> θ = { 0., 0., 0., 0.};
    vector <double> φ  = { 0., 0.};
    double w = 0.;
        
    auto startTime = SET_TIME;
    CEMV data(α, ηθ, ηφ, x0, z, T, dt, λ, M, N, ρ, σ);
    data.EMV(θ, φ, w);
    auto endTime  = SET_TIME;
    DURATION<float> totalTime = endTime - startTime;
    // cout << totalTime.count() << endl;
    return 0;
}
