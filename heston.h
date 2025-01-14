#ifndef HESTON_H
#define HESTON_H


double heston_monte_carlo_pricer(
    double S0, double K, double T, double r, 
    double kappa, double theta, double sigma, double rho, double v0,  unsigned seed,
    int n_paths, int n_steps);
#endif
