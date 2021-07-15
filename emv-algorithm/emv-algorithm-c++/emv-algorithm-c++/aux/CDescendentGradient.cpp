//
//  CDescendentGradient.cpp
//  emv-algorithm-c++
//
//  Created by Rubén Fernández Fuertes on 13/7/21.
//

#include "CDescendentGradient.hpp"
#include "PrintMacros.hpp"

SDA::CDescendentGradient(double ηθ, double ηφ, double z, double λ, vector<double> &θ, vector<double> &φ, double w, vector<vector<double>> &D)
{
    m_ηθ    = ηθ;
    m_ηφ    = ηφ;
    m_z     = z;
    m_λ     = λ;
    m_φ     = φ;
    m_θ     = θ;
    m_Oldφ1 = m_φ[0];
    m_Oldφ2 = m_φ[1];
    m_Oldθ0 = m_θ[0];
    m_Oldθ1 = m_θ[1];
    m_Oldθ2 = m_θ[2];
    m_Oldθ3 = m_θ[3];
    m_φ1    = m_Oldφ1;
    m_φ2    = m_Oldφ2;
    m_θ0    = m_Oldθ0;
    m_θ1    = m_Oldθ1;
    m_θ2    = m_Oldθ2;
    m_θ3    = m_Oldθ3;
    m_w     = w;
    m_D     = D;
    m_T     = D.back()[0];
    m_dt    = D[1][0] - D[0][0];
    m_N     = D.size() - 1;
    m_finalStep = int(floor(m_T / m_dt));
}

// UPDATES

void
SDA::updateθ1()
{
    m_θ1 = m_Oldθ1 - m_ηθ * gradientθ1();
    
}
void
SDA::updateθ2()
{
    m_θ2 = m_Oldθ2 - m_ηθ * gradientθ2();
}

void
SDA::updateθ0()
{
    m_θ0 = - m_θ2 * m_T * m_T - m_θ1 * m_T - (m_w - m_z) * (m_w - m_z);
}

void
SDA::updateφ1()
{
    m_φ1 = m_Oldφ1 - m_ηφ * gradientφ1();
    // PRINT_DATA_LINE("Old:", m_Oldφ1, "... ", "∂_{θ1}C:", gradientφ1(), "... ", "We substract:", m_ηφ * gradientφ1(),"... ", "φ:", m_φ1);
}

void
SDA::updateφ2()
{
    // cout << "Before: " << m_φ2 << ", " << gradientφ2() << endl;
    m_φ2 = m_Oldφ2 - m_ηφ * gradientφ2();
    //PRINT_DATA_LINE("Gradient:", gradientφ2(), "... ", "We substract:", m_ηφ * gradientφ2(),"... ", "φ:", m_φ2);
}

void
SDA::updateθ3()
{
    m_θ3 = 2. * m_φ2;
}

void
SDA::updateAll()
{
    // Gradient Descent
    updateθ1();
    updateθ2();
    updateφ1();
    updateφ2();
    // Depending on previous parameters
    updateθ3();
    updateθ0();
    // Keep them
    m_θ = { m_θ0, m_θ1, m_θ2, m_θ3 };
    m_φ = { m_φ1, m_φ2 };
}

// GRADIENT FUNCTIONS AND MORE STUFF

double
SDA::gradientθ1()
{
    double diffVi, t, sum = 0.;
    for (int i = 0; i < m_N; i++)
    {
        diffVi  = diffV(i);
        t       = m_D[i][0];
        sum += (diffVi - m_λ * (m_Oldφ1 + m_Oldφ2 * (m_T - t))) * m_dt;
    }
    return sum;
}

double
SDA::gradientθ2()
{
    double diffVi, t, nextt, sum = 0.;
    for (int i = 0; i < m_N; i++)
    {
        diffVi  = diffV(i);
        t       = m_D[i][0];
        nextt   = m_D[i+1][0];
        sum += (diffVi - m_λ * (m_Oldφ1 + m_Oldφ2 * (m_T - t))) * ( nextt * nextt - t * t );
    }
    return sum;
}

double
SDA::gradientφ1()
{
    double diffVi, t, sum = 0.;
    for (int i = 0; i < m_N; i++)
    {
        diffVi  = diffV(i);
        t       = m_D[i][0];
        sum += (diffVi - m_λ * (m_Oldφ1 + m_Oldφ2 * (m_T - t))) * m_dt;
    }
    return - m_λ * sum;
}

double
SDA::gradientφ2()
{
    double diffVi, t, x, NEXTt, NEXTx, sum = 0.;
    cout << m_D.size() << endl;
    for (int i = 0; i < m_N; i++)
    {
        diffVi  = diffV(i);
        t       = m_D[i][0];
        x       = m_D[i][1];
        NEXTt   = m_D[i+1][0];
        NEXTx   = m_D[i+1][1];
//        PRINT_DATA_LINE("x", x, "NEXTx", NEXTx);
        double firstFactor  =  (diffVi - m_λ * (m_Oldφ1 + m_Oldφ2 * (m_T - t))) * m_dt;
        double secondFactorNumFirstSum  = (NEXTx - m_w) * (NEXTx - m_w) * exp(-2. * m_Oldφ2 * (m_T - NEXTt)) * (m_T - NEXTt);
        double secondFactorNumSecondSum = (x - m_w) * (x - m_w) * exp(-2. * m_Oldφ2 * (m_T - t)) * (m_T - t);
        double secondFactorNum = secondFactorNumFirstSum - secondFactorNumSecondSum;
        double secondFactor = -2. * secondFactorNum / m_dt - m_λ * (m_T - t);
//        PRINT_DATA_LINE("Approximate Derivative:", diffVi, "... ", "Relax. Term:", m_λ * (m_Oldφ1 + m_Oldφ2 * (m_T - t)));
        // cout << sum << endl;
        sum += firstFactor * secondFactor;
    }
    return sum;
}

double
SDA::V(double t, double x)
{
    double V;
    double first_term  = (x - m_w) * (x - m_w) * exp( -m_Oldθ3 * (m_T - t));
    double second_term = m_Oldθ2 * t * t;
    double third_term  = m_Oldθ1 * t;
    double fourth_term = m_Oldθ0;
    V = first_term + second_term + third_term + fourth_term;
    // PRINT_DATA_LINE((x - m_w), (x - m_w)* (x - m_w), exp( -m_Oldθ3 * (m_T - t)));
    // RPRINT_DATA_LINE("θ3", m_Oldθ3, "θ2", m_Oldθ2, "θ1", m_Oldθ1, "θ0", m_Oldθ0);
    // PRINT_DATA_LINE("1st Term:", first_term, "2nd Term:", second_term, "3th Term:", third_term, "4th Term:", fourth_term);
    // PRINT_DATA_LINE("###");
    // PRINT_DATA_LINE(m_θ3, exp( -m_θ3 * (m_T - t)));
    // cout << t << ", " << m_θ1 << "," endl;
    return V;
}

double
SDA::diffV(int i)
{
    double t, x, NEXTt, NEXTx;
    double diffNum, diffVi;
    // i --> (ti, xi)
    t        = m_D[i][0];
    x        = m_D[i][1];
    // i+1 --> (t_{i+1}, x_{i+1})
    NEXTt    = m_D[i+1][0];
    NEXTx    = m_D[i+1][1];
    // Numerator
    diffNum = V(NEXTt, NEXTx) - V(t, x);
    // Aproximate derivative
    diffVi  = diffNum / m_dt;
    // PRINT_DATA_LINE("State:", x, "... ", "Next State:", NEXTx, "... ", "Value Difference:", diffNum);
    // PRINT_DATA_LINE("############");
    return diffVi;
}
