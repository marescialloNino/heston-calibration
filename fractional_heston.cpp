#include <cmath>
#include <random>
#include <chrono>
#include "fractional_heston.h"

const double PI = acos(-1);

// Function to generate correlated Gaussian random numbers
std::vector<std::vector<double>> gauss(int n_steps, double correlation) {
    std::vector<std::vector<double>> Z(2, std::vector<double>(n_steps, 0.0));
    
    // Setup random number generation
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    
    // Generate random numbers
    std::vector<std::vector<double>> u(2, std::vector<double>(n_steps));
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < n_steps; j++) {
            u[i][j] = uniform(gen);
        }
    }
    
    // Box-Muller transform
    for(int j = 0; j < n_steps; j++) {
        Z[0][j] = sqrt(-2.0 * log(u[0][j])) * cos(2.0 * PI * u[1][j]);
        Z[1][j] = sqrt(-2.0 * log(u[0][j])) * sin(2.0 * PI * u[1][j]);
    }
    
    // Apply correlation
    for(int j = 0; j < n_steps; j++) {
        Z[1][j] = correlation * Z[0][j] + sqrt(1.0 - correlation * correlation) * Z[1][j];
    }
    
    return Z;
}

// Function to build covariance matrix
std::vector<std::vector<double>> build_covariance_matrix(double sigma, double hurst, int n_steps) {
    std::vector<std::vector<double>> cov(n_steps, std::vector<double>(n_steps));
    
    for(int i = 0; i < n_steps; i++) {
        for(int j = 0; j < n_steps; j++) {
            double i_term = pow(i + 1.0, 2.0 * hurst);
            double j_term = pow(j + 1.0, 2.0 * hurst);
            double diff_term = pow(abs(i - j), 2.0 * hurst);
            cov[i][j] = (i_term + j_term - diff_term) * sigma * sigma * 0.5;
        }
    }
    
    return cov;
}

// Cholesky decomposition
std::vector<std::vector<double>> cholesky_decomposition(const std::vector<std::vector<double>>& M) {
    int n = M.size();
    std::vector<std::vector<double>> cholesky(n, std::vector<double>(n, 0.0));
    
    // First column
    cholesky[0][0] = sqrt(M[0][0]);
    for(int i = 1; i < n; i++) {
        cholesky[i][0] = M[i][0] / cholesky[0][0];
    }
    
    // For each other column
    for(int i = 1; i < n; i++) {
        // Diagonal element
        double sum = 0.0;
        for(int j = 0; j < i; j++) {
            sum += cholesky[i][j] * cholesky[i][j];
        }
        cholesky[i][i] = sqrt(M[i][i] - sum);
        
        // Rest of the column
        if(i + 1 < n) {
            for(int j = i + 1; j < n; j++) {
                sum = 0.0;
                for(int p = 0; p < i; p++) {
                    sum += cholesky[i][p] * cholesky[j][p];
                }
                cholesky[j][i] = (M[j][i] - sum) / cholesky[i][i];
            }
        }
    }
    
    return cholesky;
}

// Generate fractional Brownian motion path
std::vector<double> generate_fBM_path(double volatility, double hurst, int n_steps) {
    std::vector<double> Y(n_steps, 0.0);
    
    // Build covariance matrix and get its Cholesky decomposition
    auto cov = build_covariance_matrix(volatility, hurst, n_steps - 1);
    auto correlation = cholesky_decomposition(cov);
    
    // Generate standard normal random variables
    auto Z = gauss(n_steps - 1, 0.0);
    
    // Matrix multiplication: correlation * Z
    for(int i = 1; i < n_steps; i++) {
        Y[i] = 0.0;
        for(int j = 0; j < i; j++) {
            Y[i] += correlation[i-1][j] * Z[0][j];
        }
    }
    
    return Y;
}

// Main pricing function
std::pair<double, double> fractional_heston_price(
    double spot, double strike, double maturity, double r,
    const FractionalHestonParams& params,
    int n_sim, int n_steps) {
    
    // Initialize arrays for prices and variances
    std::vector<std::vector<double>> prices(n_sim, std::vector<double>(n_steps + 1, spot));
    std::vector<std::vector<double>> variances(n_sim, std::vector<double>(n_steps + 1, params.v0));
    
    double dt = maturity / n_steps;
    
    // Generate fBM and standard normal paths for all simulations
    std::vector<std::vector<double>> fBM(n_sim);
    std::vector<std::vector<double>> Z(n_sim);
    
    for(int n = 0; n < n_sim; n++) {
        fBM[n] = generate_fBM_path(params.v0, params.H, n_steps);
        Z[n] = gauss(n_steps, 0.0)[0];  // Take first row of gauss output
    }
    
    // Simulation
    for(int t = 1; t <= n_steps; t++) {
        for(int n = 0; n < n_sim; n++) {
            // Store previous variance for the price update
            double prev_v = variances[n][t-1];
            
            // Update variance (with absolute value to ensure positivity)
            variances[n][t] = std::abs(
                prev_v + 
                params.kappa * (params.theta - prev_v) * dt + 
                params.vol_of_vol * sqrt(prev_v * dt) * fBM[n][t-1]
            );
            
            // Update price
            prices[n][t] = prices[n][t-1] * exp(
                (r - 0.5 * prev_v) * dt + 
                sqrt(prev_v * dt) * Z[n][t-1]
            );
        }
    }
    
    // Compute payoffs at maturity
    std::vector<double> payoffs(n_sim);
    double sum_payoff = 0.0;
    double sum_squared_payoff = 0.0;
    
    for(int n = 0; n < n_sim; n++) {
        payoffs[n] = std::max(prices[n][n_steps] - strike, 0.0);
        sum_payoff += payoffs[n];
        sum_squared_payoff += payoffs[n] * payoffs[n];
    }
    
    // Calculate price and standard deviation
    double average_payoff = sum_payoff / n_sim;
    double price = exp(-r * maturity) * average_payoff;
    
    double variance = (sum_squared_payoff / n_sim - average_payoff * average_payoff);
    double std_dev = exp(-2.0 * r * maturity) * sqrt(variance / n_sim);
    
    return {price, std_dev};
} 