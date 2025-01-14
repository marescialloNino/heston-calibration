#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include "heston_fft.h"
#include "pso.h"

int main() {
    // Market data
    double S0 = 100.0;  // spot
    double r = 0.002;   // risk-free rate
    double T = 1.0;     // maturity

    // Market prices for different strikes
    std::vector<double> strikes = {95, 96, 97, 98, 99, 100, 101, 102, 103, 104};
    std::vector<double> market_prices = {
        10.93, 9.55, 8.28, 7.4, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46
    };

    // Market prices for ATM strikes
    std::vector<double> atm_strikes = {99, 100, 101};
    std::vector<double> atm_market_prices = {6.86, 6.58, 6.52};

    // Run PSO calibration
    std::cout << "Starting PSO calibration...\n\n";
    double optimal_params[6];  // kappa, theta, sigma, rho, v0, lambda
    
    particle_swarm_optimization(S0, T, r, 
                              atm_strikes.data(), atm_market_prices.data(), atm_strikes.size(),
                              optimal_params);

    // Extract calibrated parameters in correct order
    double v0 = optimal_params[0];      // initial variance
    double kappa = optimal_params[1];    // mean reversion
    double theta = optimal_params[2];    // long-term variance
    double sigma = optimal_params[3];    // vol of vol
    double rho = optimal_params[4];      // correlation
    double lambda = optimal_params[5];   // risk premium

    // Print calibrated parameters
    std::cout << "\nCalibrated Parameters:\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "v0 = " << v0 << " (initial variance)\n";
    std::cout << "kappa = " << kappa << " (mean reversion speed)\n";
    std::cout << "theta = " << theta << " (long-run variance)\n";
    std::cout << "sigma = " << sigma << " (volatility of volatility)\n";
    std::cout << "rho = " << rho << " (correlation)\n";
    std::cout << "lambda = " << lambda << " (risk premium)\n\n";

    // Compare model prices with market prices
    std::cout << "Model vs Market Prices:\n";
    std::cout << "Strike    Market    Model     Abs.Diff    Rel.Diff(%)\n";
    std::cout << "--------------------------------------------------------\n";

    double total_abs_error = 0.0;
    double total_rel_error = 0.0;
    
    for (size_t i = 0; i < strikes.size(); i++) {
        double model_price = heston_price(
            S0, strikes[i], T, r, v0, kappa, theta, sigma, rho, lambda
        );
        
        double abs_diff = std::abs(model_price - market_prices[i]);
        double rel_diff = 100.0 * abs_diff / market_prices[i];
        
        total_abs_error += abs_diff;
        total_rel_error += rel_diff;
        
        std::cout << strikes[i] << "\t" 
                  << market_prices[i] << "\t"
                  << model_price << "\t"
                  << abs_diff << "\t"
                  << rel_diff << "%\n";
    }

    // Print average errors
    std::cout << "\nAverage Absolute Error: " 
              << total_abs_error / strikes.size() << "\n";
    std::cout << "Average Relative Error: " 
              << total_rel_error / strikes.size() << "%\n";

    // Create fine grid of strikes and calculate prices
    std::cout << "\nCalculating prices for fine strike grid...\n";
    
    const double K_min = 80.0;
    const double K_max = 120.0;
    const double K_step = 0.1;
    int n_strikes = static_cast<int>((K_max - K_min) / K_step) + 1;
    
    // Open file for writing
    std::ofstream outfile("heston_prices.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "Strike,Price\n";  // CSV header
    
    // Calculate and save prices
    for (int i = 0; i < n_strikes; i++) {
        double K = K_min + i * K_step;
        double price = heston_price(
            S0, K, T, r, v0, kappa, theta, sigma, rho, lambda
        );
        
        outfile << K << "," << price << "\n";
        
        // Print progress every 10%
        if (i % (n_strikes/10) == 0) {
            std::cout << "Progress: " << (100.0 * i / n_strikes) << "%\n";
        }
    }
    
    outfile.close();
    std::cout << "Prices saved to heston_prices.txt\n";

    return 0;
}
