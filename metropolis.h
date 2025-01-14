#ifndef METROPOLIS_H
#define METROPOLIS_H

#include <vector>

// Function to generate samples using Metropolis-Hastings
std::vector<double> metropolis(int numIter);

// Uniform random number generator
double uniform_random(double mm, double nn);

// Function to generate correlated normal variables
std::pair<std::vector<double>, std::vector<double> > generate_correlated_normals_metropolis(int n_samples, double rho);

#endif
