#ifndef FRACTIONAL_HESTON_H
#define FRACTIONAL_HESTON_H

#include <vector>
#include <complex>

using cd = std::complex<double>;

// Generate fractional Brownian motion path
std::vector<double> generate_fbm(int n_steps, double H, double T, unsigned seed);

// Fractional Heston Monte Carlo pricer
double fractional_heston_monte_carlo(
    double S0, double K, double T, double r,
    double kappa, double theta, double sigma, double H, double v0,
    unsigned seed, int n_paths = 10000, int n_steps = 252);

// Characteristic function for fractional Heston
cd fractional_heston_cf(cd phi, double S0, double v0, double kappa, double theta,
                       double sigma, double H, double T, double r);

// FFT pricing for fractional Heston
double fractional_heston_price(
    double S0, double K, double T, double r, double v0,
    double kappa, double theta, double sigma, double H);

#endif 