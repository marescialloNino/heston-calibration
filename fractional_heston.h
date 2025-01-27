#ifndef FRACTIONAL_HESTON_H
#define FRACTIONAL_HESTON_H

#include <vector>
#include <random>

// Structure to hold Heston parameters including Hurst parameter
struct FractionalHestonParams {
    double v0;           // initial variance
    double kappa;        // mean reversion speed
    double theta;        // long-term variance
    double vol_of_vol;   // volatility of variance
    double correlation;  // correlation
    double H;           // Hurst parameter
};

// Function to generate correlated Gaussian random numbers
std::vector<std::vector<double>> gauss(int n_steps, double correlation);

// Function to build covariance matrix
std::vector<std::vector<double>> build_covariance_matrix(double sigma, double hurst, int n_steps);

// Cholesky decomposition
std::vector<std::vector<double>> cholesky_decomposition(const std::vector<std::vector<double>>& M);

// Generate fractional Brownian motion path
std::vector<double> generate_fBM_path(double volatility, double hurst, int n_steps);

// Main pricing function
std::pair<double, double> fractional_heston_price(
    double spot, double strike, double maturity, double r,
    const FractionalHestonParams& params,
    int n_sim = 500, int n_steps = 70
);

// Cost function for Fractional Heston
double compute_fractional_mse(
    double S0, double T, double r,
    const double* strikes, const double* market_prices, int n_options,
    double v0, double kappa, double theta, double sigma, double rho, double H
);

#endif 