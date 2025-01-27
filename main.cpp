#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include "heston_fft.h"
#include "fractional_heston.h"
#include "pso.h"

// Cost function for Fractional Heston
double compute_fractional_mse(double S0, double T, double r, 
                            const double* strikes, const double* market_prices, int n_options,
                            double v0, double kappa, double theta, double sigma, double rho, double H) {
    double mse = 0.0;
    
    FractionalHestonParams params = {
        v0,      // v0
        kappa,   // kappa
        theta,   // theta
        sigma,   // vol_of_vol
        rho,     // correlation
        H        // Hurst parameter
    };
    
    for (int i = 0; i < n_options; i++) {
        auto [model_price, std_dev] = fractional_heston_price(
            S0, strikes[i], T, r, params, 500, 70  // Using default n_sim and n_steps
        );
        double error = model_price - market_prices[i];
        mse += error * error;
    }
    
    return mse / n_options;
}

int main() {
    // Market data
    double S0 = 100.0;    // spot price
    double r = 0.002;     // risk-free rate
    double T = 1.0;       // maturity

    // Market prices for different strikes
    std::vector<double> strikes = {95, 96, 97, 98, 99, 100, 101, 102, 103, 104};
    std::vector<double> market_prices = {
        10.93, 9.55, 8.28, 7.4, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46
    };

    // Market prices for different strikes
    std::vector<double> atm_strikes = {99, 100, 101};
    std::vector<double> atm_market_prices = {
        6.86, 6.58, 6.52
    };

    // Choose model type
    bool use_fractional = true;  // Set to true for Fractional Heston
    
    std::cout << "Starting Particle Swarm Optimization for " 
              << (use_fractional ? "Fractional" : "Standard") << " Heston Model...\n\n";
    
    double optimal_params[6];  // Will hold [v0, kappa, theta, sigma, rho, lambda/H]
    particle_swarm_optimization(
        S0, T, r,
        strikes.data(), market_prices.data(), strikes.size(),
        optimal_params,
        use_fractional  // Pass flag to PSO
    );

    // Print results
    std::cout << "\nOptimization Results:\n";
    std::cout << "----------------------------------------\n";
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "Optimal Parameters:\n";
    std::cout << "v0 = " << optimal_params[0] << " (initial variance)\n";
    std::cout << "kappa = " << optimal_params[1] << " (mean reversion speed)\n";
    std::cout << "theta = " << optimal_params[2] << " (long-term variance)\n";
    std::cout << "sigma = " << optimal_params[3] << " (volatility of variance)\n";
    std::cout << "rho = " << optimal_params[4] << " (correlation)\n";
    if (use_fractional) {
        std::cout << "H = " << optimal_params[5] << " (Hurst parameter)\n\n";
    } else {
        std::cout << "lambda = " << optimal_params[5] << " (price of volatility risk)\n\n";
    }

    // Compare fitted prices with market prices
    std::cout << "Price Comparison:\n";
    std::cout << "Strike    Market    Model     Diff\n";
    std::cout << "----------------------------------------\n";
    
    double total_mse = 0.0;
    for (size_t i = 0; i < strikes.size(); i++) {
        double model_price;
        if (use_fractional) {
            FractionalHestonParams params = {
                optimal_params[0], optimal_params[1], optimal_params[2],
                optimal_params[3], optimal_params[4], optimal_params[5]
            };
            auto [price, std_dev] = fractional_heston_price(S0, strikes[i], T, r, params);
            model_price = price;
        } else {
            model_price = heston_price(
                S0, strikes[i], T, r,
                optimal_params[0], optimal_params[1], optimal_params[2],
                optimal_params[3], optimal_params[4], optimal_params[5],
                800
            );
        }
        
        double diff = std::abs((model_price - market_prices[i]) / market_prices[i]);
        total_mse += diff * diff;
        
        std::cout << strikes[i] << "\t"
                  << market_prices[i] << "\t"
                  << model_price << "\t"
                  << diff << "\n";
    }
    
    std::cout << "\nFinal MSE: " << total_mse/strikes.size() << "\n";

    // Generate 100 strike prices between 90 and 110
    std::vector<double> model_strikes(100);
    double min_strike = 90.0;
    double max_strike = 110.0;
    double step = (max_strike - min_strike) / (model_strikes.size() - 1);
    
    for (size_t i = 0; i < model_strikes.size(); ++i) {
        model_strikes[i] = min_strike + i * step;
    }

    // Open a file to write the strike prices and model prices
    std::ofstream results_file("strike_prices.txt");
    if (!results_file.is_open()) {
        std::cerr << "Error opening file for writing results." << std::endl;
        return 1;
    }

    // Calculate and write model prices for each strike
    for (double strike : model_strikes) {
        double model_price;
        if (use_fractional) {
            FractionalHestonParams params = {
                optimal_params[0], optimal_params[1], optimal_params[2],
                optimal_params[3], optimal_params[4], optimal_params[5]
            };
            auto [price, std_dev] = fractional_heston_price(S0, strike, T, r, params);
            model_price = price;
        } else {
            model_price = heston_price(
                S0, strike, T, r, 
                optimal_params[0], optimal_params[1], 
                optimal_params[2], optimal_params[3], 
                optimal_params[4], optimal_params[5], 
                800
            );
        }
        results_file << strike << "," << model_price << "\n"; // Write strike and model price
    }

    // Close the results file
    results_file.close();

    return 0;
}
