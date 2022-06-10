#ifndef __CUDA_PROBABILITIES_H
#define __CUDA_PROBABILITIES_H
#include <complex>
#include <cuComplex.h>


// GPU Standard Oscilation
void cuda_StandardOscilation(
	cuDoubleComplex* _U, double* _energy, int _size_data, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double* _events ,cuDoubleComplex** _batchedDM);

void cuda_simulation_StandardOscilation(
	int num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho, 
	double* _dm, double* _alpha);

// GPU Invisible Decay
void cuda_InvisibleDecay(
	cuDoubleComplex* _U, double* _energy, int _size_data, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double* _events, cuDoubleComplex** _batchedDM);

class ProbConst {
	public:
		ProbConst(){}
		static const std::complex<double> I;
		static const std::complex<double> Z0;
		static const double hbar;
		static const double clight;
		//double GevkmToevsq = hbar*clight*1.e15;
		static const double GevkmToevsq;
};

// Matriz U
cuDoubleComplex* make_umns(int sg, double* th, double dcp);
// Matriz H: hamiltoniano
// int hamilton(int q, double Eni);
// double dGdE(double mi, double mf, double Ei, double Ef, int coup);
// double dGbdE(double mi, double mf, double Ei, double Ef, int coup);

#endif