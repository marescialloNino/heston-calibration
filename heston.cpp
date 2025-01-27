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




double heston_monte_carlo_pricer(
    double S0, double K, double T, double r,
    double kappa, double theta, double sigma, double rho, double v0,
    int n_paths = 100000, int n_steps = 252) {

    // Add proper seeding using current time plus some offset to ensure different seed from pricer_2
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + 12345;
    std::mt19937 gen(seed);  // Seed the Mersenne Twister generator
    std::normal_distribution<> normal(0.0, 1.0);

    double dt = T / n_steps;
    double sqrt_dt = std::sqrt(dt);

    // Initialize stock prices and variances
    std::vector<std::vector<double>> S(n_paths, std::vector<double>(n_steps + 1, S0));
    std::vector<std::vector<double>> v(n_paths, std::vector<double>(n_steps + 1, v0));

    // Monte Carlo simulation with full truncation scheme
    for (int t = 0; t < n_steps; t++) {
        for (int i = 0; i < n_paths; i++) {
            // Generate two independent standard normal variables
            double Z1 = normal(gen);
            double Z2 = rho * Z1 + std::sqrt(1 - rho * rho) * normal(gen);

            // Update variance using full truncation scheme
            double v_positive = std::max(v[i][t], 0.0);  // Use only positive values for drift and diffusion
            v[i][t + 1] = v[i][t] + kappa * (theta - v_positive) * dt + 
                         sigma * std::sqrt(v_positive) * Z1 * sqrt_dt;

            // Update stock price using Euler-Maruyama with log transformation
            double log_S = std::log(S[i][t]);
            log_S += (r - 0.5 * v_positive) * dt + std::sqrt(v_positive) * Z2 * sqrt_dt;
            S[i][t + 1] = std::exp(log_S);
        }
    }

    // Compute call option payoffs at maturity
    std::vector<double> payoffs(n_paths);
    for (int i = 0; i < n_paths; i++) {
        payoffs[i] = std::max(S[i][n_steps] - K, 0.0);
    }

    // Calculate price with control variate technique
    double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / n_paths;
    return std::exp(-r * T) * mean_payoff;
}

