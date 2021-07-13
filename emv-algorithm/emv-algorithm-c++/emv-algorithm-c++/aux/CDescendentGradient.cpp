//
//  CDescendentGradient.cpp
//  emv-algorithm-c++
//
//  Created by Rubén Fernández Fuertes on 13/7/21.
//

#include "CDescendentGradient.hpp"

SDA::CDescendentGradient(double ηθ, double ηφ, double z, double λ, vector<double> &φ, vector<double> &θ, double w, vector<vector<double>> &D)
{
    m_ηθ    = ηθ;
    m_ηφ    = ηφ;
    m_z     = z;
    m_λ     = λ;
    m_φ     = φ;
    m_θ     = θ;
    m_φ1    = φ[0];
    m_φ2    = φ[1];
    m_θ0    = θ[0];
    m_θ1    = θ[1];
    m_θ2    = θ[2];
    m_θ3    = θ[3];
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
    m_θ1 -= m_ηθ * gradientθ1();
}
void
SDA::updateθ2()
{
    m_θ2 -= m_ηθ * gradientθ2();
}

void
SDA::updateθ0()
{
    m_θ0 = - m_θ2 * m_T * m_T - m_θ1 * m_T - (m_w - m_z) * (m_w - m_z);
}

void
SDA::updateφ1()
{
    m_φ1 -= m_ηφ * gradientφ1();
}

void
SDA::updateφ2()
{
    m_φ2 -= m_ηφ * gradientφ2();
}

void
SDA::updateθ3()
{
    m_θ2 = 2. * m_φ2;
}

void
SDA::updateAll()
{
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
        sum += (diffVi - m_λ * (m_φ1 + m_φ2 * (m_T - t))) * m_dt;
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
        sum += (diffVi - m_λ * (m_φ1 + m_φ2 * (m_T - t))) * ( nextt * nextt - t * t );
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
        sum += (diffVi - m_λ * (m_φ1 + m_φ2 * (m_T - t))) * m_dt;
    }
    return - m_λ * sum;
}

double
SDA::gradientφ2()
{
    double diffVi, t, x, NEXTt, NEXTx, sum = 0.;
    for (int i = 0; i < m_N; i++)
    {
        diffVi  = diffV(i);
        t       = m_D[i][0];
        x       = m_D[i][1];
        NEXTt   = m_D[i][0];
        NEXTx   = m_D[i+1][1];
        double firstFactor  =  (diffVi - m_λ * (m_φ1 + m_φ2 * (m_T - t))) * m_dt;
        double secondFactorNumFirstSum  = (NEXTx - m_w) * (NEXTx - m_w) * exp(-2. * m_φ2 * (m_T - NEXTt)) * (m_T - NEXTt);
        double secondFactorNumSecondSum = (x - m_w) * (x - m_w) * exp(-2. * m_φ2 * (m_T - t)) * (m_T - t);
        double secondFactorNum = secondFactorNumFirstSum - secondFactorNumSecondSum;
        double secondFactor = -2. * secondFactorNum / m_dt - m_λ * (m_T - t);
        sum += firstFactor * secondFactor;
    }
    return sum;
}

double
SDA::V(double t, double x)
{
    double first_term  = (x - m_w) * (x - m_w) * exp( -m_θ3 * (m_T - t));
    double second_term = m_θ2 * t * t;
    double third_term  = m_θ1 * t;
    double fourth_term = m_θ0;
    return first_term + second_term + third_term + fourth_term;
}

double
SDA::diffV(int i)
{
    // i --> (ti, xi)
    double Time         = m_D[i][0];
    double State        = m_D[i][1];
    // i+1 --> (t_{i+1}, x_{i+1})
    double nextTime     = m_D[i+1][0];
    double nextState    = m_D[i+1][1];
    // Numerator
    double diffNum = V(nextTime, nextState) - V(Time, State);
    return diffNum / m_dt;
}
