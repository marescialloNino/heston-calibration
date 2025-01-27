#ifndef HESTON_PRICER_H
#define HESTON_PRICER_H

#include <complex>

using cd = std::complex<double>;

// Characteristic function
cd heston_cf(cd phi, double S0, double v0, double kappa, double theta,
             double sigma, double rho, double lambda, double tau, double r);

// Integrand function
cd integrand(double phi, double S0, double K, double T, double r, double v0,
            double kappa, double theta, double sigma, double rho, double lambda);

// Direct integration pricer (quadrature method)
double heston_price(double S0, double K, double T, double r, double v0,
                   double kappa, double theta, double sigma, double rho,
                   double lambda, int N=1000);

#endif
