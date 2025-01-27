#ifndef PSO_H
#define PSO_H

void particle_swarm_optimization(double S0, double T, double r,
                               const double* strikes, const double* market_prices, int n_options,
                               double* optimal_params,
                               bool use_fractional = false);

#endif
