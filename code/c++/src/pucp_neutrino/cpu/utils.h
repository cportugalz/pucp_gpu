#ifndef __VIS_H
#define __VIS_H
#include <complex>
#include <eigen3/Eigen/Eigen>

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

double square(double x);
double min(double x, double y);
// Matriz U
std::complex<double>** make_umns(int sg, double* th, double dcp);
// Matriz H: hamiltoniano
// int hamilton(int q, double Eni);
double dGdE(double mi, double mf, double Ei, double Ef, int coup);
double dGbdE(double mi, double mf, double Ei, double Ef, int coup);
double IntegrandoPhi(double _ei, double* _mss, Eigen::Matrix3cd _Pot, 
	double _enf, double _l, double* _th, double _d, 
	double* _alpha, int _tfi, int _tsi, int _tff, int _tsf, 
	int _tpar, int _thij, int _tqcoup);

#endif