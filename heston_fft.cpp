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

// Price using numerical integration 
double heston_price(double S0, double K, double T, double r, double v0,
                   double kappa, double theta, double sigma, double rho, double lambda) {
    const int N = 200;  
    const double umax = 100.0;
    const double dphi = umax / N;
    
    // Precompute constants that don't change in the loop
    const cd I_complex(0.0, 1.0);
    const double exp_rT = exp(r * T);
    const double exp_mrt = exp(-r * T);
    
    // Precompute common terms for characteristic function
    const cd a = kappa * theta;
    const cd b = kappa + lambda;
    const double sigma_sq = sigma * sigma;
    const double sigma_sq_inv = 1.0 / sigma_sq;
    const cd two_a_sigma_sq_inv = -2.0 * a * sigma_sq_inv;
    
    cd P(0.0, 0.0);
    
    for (int i = 1; i < N; i++) {
        double phi = dphi * (2.0 * i + 1.0) / 2.0;
        
        // Compute characteristic function for phi-i
        cd phi_minus_i(phi, -1.0);
        cd rspi1 = rho * sigma * phi_minus_i * I_complex;
        cd b_minus_rspi1 = b - rspi1;
        cd d1 = sqrt(b_minus_rspi1 * b_minus_rspi1 + 
                    (phi_minus_i * I_complex + phi_minus_i * phi_minus_i) * sigma_sq);
        cd g1 = (b_minus_rspi1 + d1) / (b_minus_rspi1 - d1);
        
        cd exp_d1T = exp(d1 * T);
        cd g1_exp_d1T = g1 * exp_d1T;
        cd one_minus_g1 = 1.0 - g1;
        cd one_minus_g1_exp_d1T = 1.0 - g1_exp_d1T;
        
        cd cf1 = exp(r * phi_minus_i * I_complex * T) * 
                 pow(S0, phi_minus_i * I_complex) * 
                 pow(one_minus_g1_exp_d1T/one_minus_g1, two_a_sigma_sq_inv) *
                 exp(a * T * (b_minus_rspi1 + d1) * sigma_sq_inv + 
                     v0 * (b_minus_rspi1 + d1) * (1.0 - exp_d1T) / 
                     (sigma_sq * one_minus_g1_exp_d1T));
        
        // Compute characteristic function for phi
        cd phi_complex(phi, 0.0);
        cd rspi2 = rho * sigma * phi_complex * I_complex;
        cd b_minus_rspi2 = b - rspi2;
        cd d2 = sqrt(b_minus_rspi2 * b_minus_rspi2 + 
                    (phi_complex * I_complex + phi_complex * phi_complex) * sigma_sq);
        cd g2 = (b_minus_rspi2 + d2) / (b_minus_rspi2 - d2);
        
        cd exp_d2T = exp(d2 * T);
        cd g2_exp_d2T = g2 * exp_d2T;
        cd one_minus_g2 = 1.0 - g2;
        cd one_minus_g2_exp_d2T = 1.0 - g2_exp_d2T;
        
        cd cf2 = exp(r * phi_complex * I_complex * T) * 
                 pow(S0, phi_complex * I_complex) * 
                 pow(one_minus_g2_exp_d2T/one_minus_g2, two_a_sigma_sq_inv) *
                 exp(a * T * (b_minus_rspi2 + d2) * sigma_sq_inv + 
                     v0 * (b_minus_rspi2 + d2) * (1.0 - exp_d2T) / 
                     (sigma_sq * one_minus_g2_exp_d2T));
        
        cd numerator = exp_rT * cf1 - K * cf2;
        cd denominator = I_complex * phi * pow(K, I_complex * phi);
        
        P += dphi * numerator / denominator;
    }
    
    return ((S0 - K * exp_mrt) / 2.0 + P.real() / PI);
}