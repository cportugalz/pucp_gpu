#ifndef __PROBABILITIES_H
#define __PROBABILITIES_H
#include <complex>

// Standard Oscilation
void StandardOscilation(
	std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double** _P );

// Invisible Decay
void InvisibleDecay(
	std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double** _P );

// Violation of Equivalence Principle
void ViolationPrincipleDecay(
	std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho,
	double* d_m, double* _gamma, double** _P );

// Non Standard Interaction
void NonStandardInteraction(
	std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho, 
	double* _dm, double* _parmNSI, double** _P );

void Probability_Vis(
	double _energy, double _L, double _rho, double* th,
	double* _dm, double d, double* _alpha, double _mlight, int _tfi, int _tsi, int _tff,
	int _tsf, int _tpar, int _thij, int _tqcoup, double* _P );

#endif