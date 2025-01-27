#include <complex>
#include <cmath>
#include "heston_fft.h"

const double PI = acos(-1);
const cd I(0.0, 1.0);

// Characteristic function
cd heston_cf(cd phi, double S0, double v0, double kappa, double theta, 
             double sigma, double rho, double lambda, double tau, double r) {
    // Constants
    cd a = kappa * theta;
    cd b = kappa + lambda;
    
    // Common terms w.r.t phi
    cd rspi = rho * sigma * phi * I;
    
    // Define d parameter
    cd d = sqrt(pow(rho * sigma * phi * I - b, 2.0) + 
               (phi * I + pow(phi, 2.0)) * pow(sigma, 2.0));
    
    // Define g parameter
    cd g = (b - rspi + d) / (b - rspi - d);
    
    // Calculate characteristic function by components
    cd exp1 = exp(r * phi * I * tau);
    cd term2 = pow(S0, phi * I) * pow((1.0 - g * exp(d * tau))/(1.0 - g), 
                                     -2.0 * a / pow(sigma, 2.0));
    cd exp2 = exp(a * tau * (b - rspi + d) / pow(sigma, 2.0) + 
                  v0 * (b - rspi + d) * (1.0 - exp(d * tau)) / 
                  (sigma * sigma * (1.0 - g * exp(d * tau))));
    
    return exp1 * term2 * exp2;
}

// Integrand function
cd integrand(double phi, double S0, double K, double T, double r, double v0,
            double kappa, double theta, double sigma, double rho, double lambda) {
    // Create phi - i and phi complex numbers
    cd phi_minus_i(phi, -1.0);
    cd phi_complex(phi, 0.0);
    
    // Calculate characteristic functions
    cd cf1 = heston_cf(phi_minus_i, S0, v0, kappa, theta, sigma, rho, lambda, T, r);
    cd cf2 = heston_cf(phi_complex, S0, v0, kappa, theta, sigma, rho, lambda, T, r);
    
    // Calculate numerator and denominator
    cd numerator = exp(r * T) * cf1 - K * cf2;
    cd denominator = I * phi * pow(K, I * phi);
    
    return numerator / denominator;
}

// Direct integration pricer (quadrature method)
double heston_price(double S0, double K, double T, double r, double v0,
                   double kappa, double theta, double sigma, double rho, double lambda, int N) {
    
    const double umax = 100.0;
    const double dphi = umax / N;
    
    cd P(0.0, 0.0);
    
    // Integration loop
    for (int i = 1; i < N; i++) {
        double phi = dphi * (2.0 * i + 1.0) / 2.0;
        P += dphi * integrand(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, lambda);
    }
    
    return (S0 - K * exp(-r * T)) / 2.0 + P.real() / PI;
}