#ifndef __PROBABILITIES_H
#define __PROBABILITIES_H
#include <complex>


// Standard Oscilation
void StandardOscilation(
	double Energ, int sigN, double L, double rhomat, double* th, 
	double dCP, double* dm, double* alpha, double** P);
// Invisible Decay
void InvisibleDecay(
	double Energ, int sigN, double L, double rhomat, double* th, 
	double dCP, double* dm, double* alpha, double** P);
// Violation of Equivalence Principle
void ViolationPrincipleDecay(
	double Energ, int sigN, double L, double rhomat, double* th,
	double dCP, double* dm, double* gamma, double** P );
// Non Standard Interaction
void NonStandardInteraction(
	double Energ, int sigN, double L, double rhomat, double* th, 
	double dCP, double* dm, double* parmNSI, double** P );


class ProbConst {
	public:
		ProbConst(){}
		static const std::complex<double> I;
		static const double hbar;
		static const double clight;
		//double GevkmToevsq = hbar*clight*1.e15;
		static const double GevkmToevsq;
};
#endif