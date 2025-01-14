#include <stdlib.h>
#include <math.h>
#include <vector>
#include "metropolis.h"
#include <chrono>

// Gaussian distribution function for standard normal
double q(double x) {
    return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * x * x); // Standard normal: mu = 0, sigma = 1
}

std::vector<double> metropolis(int numIter) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + 54321;
    srand(seed);

    std::vector<double> points(numIter);
    
    double r = 0.0;           // Initial position
    double p = q(r);          // Probability at initial position

    for (int i = 0; i < numIter; i++) {
        // Propose a new position
        double rn = r + uniform_random(-1.0, 1.0); // Random value from -1 to 1
        double pn = q(rn);

        if (pn >= p || uniform_random(0.0, 1.0) < pn / p) {
            r = rn; // Accept the proposal
            p = pn;
        }

        points[i] = r; // Store the current position
    }

    return points;
}

double uniform_random(double mm, double nn) {
    return mm + ((double)rand() / RAND_MAX) * (nn - mm); // Generate [mm, nn)
}

std::pair<std::vector<double>, std::vector<double>> generate_correlated_normals_metropolis(int n_samples, double rho) {
    // Generate independent standard normal variables
    std::vector<double> v1 = metropolis(n_samples);
    std::vector<double> v2_independent = metropolis(n_samples);

    // Generate the correlated variable
    std::vector<double> v2(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        v2[i] = rho * v1[i] + sqrt(1 - rho * rho) * v2_independent[i];
    }

    return {v1, v2};
}

