#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include "heston.h"
#include "metropolis.h"

double heston_monte_carlo_pricer(
    double S0, double K, double T, double r,
    double kappa, double theta, double sigma, double rho, double v0, unsigned seed,
    int n_paths = 100000, int n_steps = 252) {

    srand(seed);  // This affects the Metropolis-Hastings sampling

    double dt = T / n_steps; // Time step size

    // Initialize stock prices and variances
    std::vector<std::vector<double>> S(n_paths, std::vector<double>(n_steps + 1, 0.0));
    std::vector<std::vector<double>> v(n_paths, std::vector<double>(n_steps + 1, 0.0));

    // Set initial values
    for (int i = 0; i < n_paths; i++) {
        S[i][0] = S0;
        v[i][0] = v0;
    }

    // Monte Carlo simulation
    for (int t = 1; t <= n_steps; t++) {
        // Generate correlated random variables using Metropolis-Hastings
        auto [v1, v2] = generate_correlated_normals_metropolis(n_paths, rho);

        for (int i = 0; i < n_paths; i++) {
            double W_v = v1[i] * std::sqrt(dt); // Random variable for variance
            double W_S = v2[i] * std::sqrt(dt); // Correlated random variable for stock price

            // Update variance (Heston process for v_t)
            v[i][t] = v[i][t - 1] + kappa * (theta - v[i][t - 1]) * dt +
                      sigma * std::sqrt(std::abs(v[i][t - 1])) * W_v;
            v[i][t] = std::abs(v[i][t]); // Ensure non-negative variance

            // Update stock price (Heston process for S_t)
            S[i][t] = S[i][t - 1] * std::exp(
                (r - 0.5 * v[i][t-1]) * dt + std::sqrt(std::abs(v[i][t-1])) * W_S);
        }
    }

    // Compute payoffs at maturity
    std::vector<double> payoffs(n_paths, 0.0);
    for (int i = 0; i < n_paths; i++) {
        payoffs[i] = std::max(S[i][n_steps] - K, 0.0); // Payoff for each path
    }

    // Compute discounted average of payoffs
    double average_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / n_paths;
    double price = std::exp(-r * T) * average_payoff;

    return price;
}


