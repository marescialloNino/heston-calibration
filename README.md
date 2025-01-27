In this program I'm trying to implement an Heston Model Pricer and Calibration framework without using external libraries.

For the Monte Carlo Pricer, to generate two correlated sets of random variables, i use the Metropolis Hasting Algorithm.

It's not feasible to use this pricer for the optimization, since i want to utilize the particle swarm optimization algorithm and i need fast computations, so i implemented a function to compute the complex integral obtained by Heston in his paper (1993) with the method of characteristic functions with a simple quadrature formula.

I tried to implement also the fractional Heston model with relative cost function to be optimized by particle swarm, but for now it's not working, since for pricing I use a monte carlo method, which is not fast enough.
