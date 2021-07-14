//
//  CDescendentGradient.hpp
//  emv-algorithm-c++
//
//  Created by Rubén Fernández Fuertes on 13/7/21.
//

#ifndef CDescendentGradient_hpp
#define CDescendentGradient_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#define PI atan(1)*4
using namespace std;

class CDescendentGradient
{
public:
    CDescendentGradient(double ηθ, double ηφ, double z, double λ, vector<double> &θ, vector<double> &φ, double w, vector<vector<double>> &D);
    void updateAll();
    // GET VECTOR PARAMETERS
    inline vector <double> Getθ() {return m_θ;};
    inline vector <double> Getφ() {return m_φ;};
private:
    // VALUE FUNCTION AND DERIVATIVE VALUE FUNCTION APPROXIMATION
    double V(double t, double x);
    double diffV(int i);
    // UPDATE FUNCTIONS
    void updateθ0();
    void updateθ1();
    void updateθ2();
    void updateθ3();
    void updateφ1();
    void updateφ2();
    // GRADIENTS
    double gradientθ1();
    double gradientθ2();
    double gradientφ1();
    double gradientφ2();
    // VARIABLES AND PARAMETERS
    double m_ηθ;
    double m_ηφ;
    double m_z;
    double m_T;
    double m_dt;
    double m_λ;
    double m_N;
    double m_finalStep;
    vector<double> m_φ;
    vector<double> m_θ;
    double m_φ1;
    double m_φ2;
    double m_θ0;
    double m_θ1;
    double m_θ2;
    double m_θ3;
    double m_w;
    vector<vector<double>> m_D;
};

#define SDA CDescendentGradient
#define PRINT(a) cout << a << endl;

#endif /* CDescendentGradient_hpp */
