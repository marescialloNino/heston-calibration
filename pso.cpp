#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <chrono>
#include "pso.h"
#include "heston_fft.h"
#include "fractional_heston.h"


double compute_mse(double S0, double T, double r, 
                  const double* strikes, const double* market_prices, int n_options,
                  double v0, double kappa, double theta, double sigma, double rho, double lambda) {
    double mse = 0.0;
    
    for (int i = 0; i < n_options; i++) {
        double model_price = heston_price(
            S0, strikes[i], T, r, 
            v0,     // v0 (initial variance)
            kappa,  // kappa (mean reversion)
            theta,  // theta (long-term variance)
            sigma,  // sigma (vol of vol)
            rho,    // rho (correlation)
            lambda , // lambda (risk premium)
            800
        );
        double error = model_price - market_prices[i];
        mse += error * error;
    }
    
    return mse / n_options;
}

void particle_swarm_optimization(double S0, double T, double r,
                               const double* strikes, const double* market_prices, int n_options,
                               double* optimal_params,
                               bool use_fractional) {
    int n_particles = 10;     // Number of particles
    int n_iterations = 10;     // Number of iterations
    const double MAX_VELOCITY = 1.8;  // Add constant max velocity
    const int DISCOVERY_MODE_THRESHOLD = 40; // Threshold for discovery mode
    const int SUPER_DISCOVERY_MODE_THRESHOLD = 3; // Threshold for super discovery mode

    // Fix the bounds initialization
    double bounds[6][2] = {
        {0.01, 3.0},    // [0] v0: initial variance
        {0.0, 5.0},     // [1] kappa: mean reversion
        {0.0, 5.0},     // [2] theta: long-term variance
        {0.0, 5.0},     // [3] sigma: vol of vol
        {-1.0, 1.0},    // [4] rho: correlation
        {0.0, use_fractional ? 1.0 : 5.0}  // [5] H or lambda
    };

    // Initialize particles and velocities
    double** particles = new double*[n_particles];
    double** velocities = new double*[n_particles];
    double** personal_best = new double*[n_particles];
    double* personal_best_cost = new double[n_particles];
    double global_best[6];
    double global_best_cost = DBL_MAX;

    // Initialize particles with more focused initial distribution
    for (int i = 0; i < n_particles; i++) {
        particles[i] = new double[6];
        velocities[i] = new double[6];
        personal_best[i] = new double[6];
        
        for (int j = 0; j < 6; j++) {
            // More focused initialization around typical values
            double mid = (bounds[j][0] + bounds[j][1]) / 2.0;
            double range = bounds[j][1] - bounds[j][0];
            particles[i][j] = mid + ((double)rand() / RAND_MAX - 0.5) * range;
            velocities[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.5 * range;
            personal_best[i][j] = particles[i][j];
        }
        personal_best_cost[i] = DBL_MAX;
    }

    // Adaptive parameters
    double w_start = 1.0;     // Initial inertia weight
    double w_end = 0.8;       // Final inertia weight

    int no_improvement_count = 0; // Counter for iterations without improvement
    int discovery_mode_count = 0; // Counter for discovery mode activations
    double best_cost = DBL_MAX; // Track the best cost found

    // Main PSO loop
    for (int iter = 0; iter < n_iterations; iter++) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        // Adaptive inertia weight
        double w = w_start - (w_start - w_end) * iter / n_iterations;
        double cognitive_component = 2.0;
        double social_component = 1.0;
        // Update particles
        for (int i = 0; i < n_particles; i++) {
            double cost;
            if (use_fractional) {
                cost = compute_fractional_mse(S0, T, r, strikes, market_prices, n_options,
                                            particles[i][0],  // v0
                                            particles[i][1],  // kappa
                                            particles[i][2],  // theta
                                            particles[i][3],  // sigma
                                            particles[i][4],  // rho
                                            particles[i][5]); // H
            } else {
                cost = compute_mse(S0, T, r, strikes, market_prices, n_options,
                                 particles[i][0],  // v0
                                 particles[i][1],  // kappa
                                 particles[i][2],  // theta
                                 particles[i][3],  // sigma
                                 particles[i][4],  // rho
                                 particles[i][5]); // lambda
            }

            if (cost < personal_best_cost[i]) {
                personal_best_cost[i] = cost;
                for (int j = 0; j < 6; j++) {
                    personal_best[i][j] = particles[i][j];
                }
            }
            if (cost < global_best_cost) {
                global_best_cost = cost;
                for (int j = 0; j < 6; j++) {
                    global_best[j] = particles[i][j];
                }
            }
        }

        // Update velocities and positions
        for (int i = 0; i < n_particles; i++) {
            for (int j = 0; j < 6; j++) {
                double r1 = (double)rand() / RAND_MAX;
                double r2 = (double)rand() / RAND_MAX;

                velocities[i][j] = w * velocities[i][j] + 
                                 cognitive_component * r1 * (personal_best[i][j] - particles[i][j]) +
                                 social_component * r2 * (global_best[j] - particles[i][j]);

                // Use constant max velocity instead of adaptive
                velocities[i][j] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, velocities[i][j]));

                particles[i][j] += velocities[i][j];
                particles[i][j] = fmax(bounds[j][0], fmin(bounds[j][1], particles[i][j]));
            }
        }

        // Check for improvement
        if (global_best_cost < best_cost) {
            best_cost = global_best_cost;
            no_improvement_count = 0; // Reset counter on improvement
        } else {
            no_improvement_count++;
        }

        // Discovery mode: shake particles if no improvement for a threshold
        if (no_improvement_count >= DISCOVERY_MODE_THRESHOLD) {
            printf("Entering discovery mode: shaking particles around the parameter space.\n"); // Print message
            for (int i = 0; i < n_particles; i++) {
                for (int j = 0; j < 6; j++) {
                    // Shake particles around the parameter space
                    double mid = (bounds[j][0] + bounds[j][1]) / 2.0;
                    double range = bounds[j][1] - bounds[j][0];
                    particles[i][j] = mid + ((double)rand() / RAND_MAX - 0.5) * range;
                }
            }
            no_improvement_count = 0; // Reset counter after shaking
            discovery_mode_count++; // Increment discovery mode counter
        }

        // Super discovery mode: more aggressive shaking if discovery mode has been triggered multiple times
        if (discovery_mode_count >= SUPER_DISCOVERY_MODE_THRESHOLD) {
            printf("Entering super discovery mode: aggressive shaking of particles.\n"); // Print message
            for (int i = 0; i < n_particles; i++) {
                for (int j = 0; j < 6; j++) {
                    // Aggressive shaking: larger range
                    double mid = (bounds[j][0] + bounds[j][1]) / 2.0;
                    double range = (bounds[j][1] - bounds[j][0]) * 2.0; // Double the range for aggressive shaking
                    particles[i][j] = mid + ((double)rand() / RAND_MAX - 0.5) * range;
                }
            }
            discovery_mode_count = 0; // Reset counter after super discovery mode
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iter_time = iter_end - iter_start;
        
        if (iter % 10 == 0) {
            printf("Iteration %d: MSE = %.6f, Time = %.2f seconds\n", 
                   iter + 1, global_best_cost, iter_time.count());
        }
    }

    // Copy best solution
    for (int j = 0; j < 6; j++) {
        optimal_params[j] = global_best[j];
    }

    // Cleanup
    for (int i = 0; i < n_particles; i++) {
        delete[] particles[i];
        delete[] velocities[i];
        delete[] personal_best[i];
    }
    delete[] particles;
    delete[] velocities;
    delete[] personal_best;
    delete[] personal_best_cost;
}