#include "utils.h"
#include <complex>
#include <eigen3/Eigen/Eigenvalues>
#include "integrate/_1D/GaussianQuadratures/GaussLegendre.hpp"
using namespace std;

const std::complex<double> ProbConst::I = std::complex<double>(0, 1);
const std::complex<double> ProbConst::Z0 = std::complex<double>(0, 0);
const double ProbConst::hbar = 6.58211928*1.e-25;
const double ProbConst::clight = 299792458;
//double GevkmToevsq = hbar*clight*1.e15;
const double ProbConst::GevkmToevsq = 0.197327; // Approximated value


double square(double x) {
	return x * x;
}

double min(double x, double y) {
	if (x < y)
		return x;
	else
		return y;
}

// Matriz U
std::complex<double>** make_umns(int _sg, double* _th, double _dcp) {
	double th12 = _th[0];
	double th13 = _th[1];
	double th23 = _th[2];
	double delta = _sg * _dcp;
	// U matrix for calculations
	std::complex<double>** U = new std::complex<double>*[3];	
	for (int i=0; i<3; i++){
		U[i] = new std::complex<double>[3];
	}
	U[0][0] = cos(th12) * cos(th13);
	U[0][1] = sin(th12) * cos(th13);
	U[0][2] = sin(th13) * exp(-ProbConst::I * delta);
	U[1][0] = -sin(th12) * cos(th23) - cos(th12) * sin(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[1][1] = cos(th12) * cos(th23) - sin(th12) * sin(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[1][2] = sin(th23) * cos(th13);
	U[2][0] = sin(th12) * sin(th23) - cos(th12) * cos(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[2][1] = -cos(th12) * sin(th23) - sin(th12) * cos(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[2][2] = cos(th23) * cos(th13);
	return U;
}

double dGdE(double mi, double mf, double Ei, double Ef, int coup) {
	double xif = mi / mf, ei = Ei * 1.e-9, ef = Ef * 1.e-9, tmp;
	if (coup == 1) {
		if (ei / xif <= ef && ef <= ei) {
			tmp = (1. / sqrt(1 - mi / square(ei * 1.e9))) * (xif / (xif - 1)) * (1. / (ei * ef)) * (square(ei + sqrt(xif) * ef) / square(sqrt(xif) + 1));
		}
		else {
			tmp = 0;
		}
	}
	else {
		if (ei / xif <= ef && ef <= ei){
			tmp = (1. / sqrt(1 - mi / square(ei * 1.e9))) * (xif / (xif - 1)) * (1. / (ei * ef)) * (square(ei - sqrt(xif) * ef) / square(sqrt(xif) - 1));
		}
		else {
			tmp = 0;
		}
	}
	return tmp;
}

double dGbdE(double mi, double mf, double Ei, double Ef, int coup) {
	double xif = mi / mf, ei = Ei * 1.e-9, ef = Ef * 1.e-9, tmp;
	if (coup == 1) {
		if (ei / xif <= ef && ef <= ei) {
			tmp = (1. / sqrt(1 - mi / square(ei * 1.e9))) * (xif / (xif - 1)) * ((ei - ef) / (ei * ef)) * ((xif * ef - ei) / square(sqrt(xif) + 1));
		}
		else {
			tmp = 0;
		}
	}
	else {
		if (ei / xif <= ef && ef <= ei) {
			tmp = (1. / sqrt(1 - mi / square(ei * 1.e9))) * (xif / (xif - 1)) * ((ei - ef) / (ei * ef)) * ((xif * ef - ei) / square(sqrt(xif) - 1));
		}
		else {
			tmp = 0;
		}
	}
	return tmp;
}

double IntegrandoPhi(double _ei, double* _mss, Eigen::Matrix3cd _Pot, double _enf, double _l, double* _th, double _d, 
	double* _alpha, int _tfi, int _tsi, int _tff, int _tsf, int _tpar, int _thij, int _tqcoup) {
	double Eni = _ei * 1.e9;
	double ef = _enf * 1.e-9;
	double dist = _l * 1.e9 / ProbConst::GevkmToevsq;
	// Objeto para almacenar el Eigensistema
	Eigen::ComplexEigenSolver<Eigen::Matrix3cd> tmpVD;
	/* Calculamos Autovalores y Autovectores iniiales */
	Eigen::Matrix3cd Htemp;
	Eigen::Matrix3cd Utemp; //(3,3)
	std::complex<double>** _U = make_umns(_tsi, _th, _d);
	Utemp << _U[0][0], _U[0][1], _U[0][2],
		_U[1][0], _U[1][1], _U[1][2],
		_U[2][0], _U[2][1], _U[2][2];
	Htemp << std::complex<double>(0.5 * _mss[0] / Eni, -0.5 * _alpha[0] / Eni), ProbConst::Z0, ProbConst::Z0,
		ProbConst::Z0, std::complex<double>(0.5 * _mss[1] / Eni, -0.5 * _alpha[1] / Eni), ProbConst::Z0,
		ProbConst::Z0, ProbConst::Z0, std::complex<double>(0.5 * _mss[2] / Eni, -0.5 * _alpha[2] / Eni);
	/* Hamiltoniano */
	Htemp = Utemp * Htemp * Utemp.adjoint() + _tsi * _Pot;
	tmpVD.compute(Htemp);

	/* Masas y alpha tíldes iniciales */
	double massfisq[3] = {2 * Eni * real(tmpVD.eigenvalues()[0]), 2 * Eni * real(tmpVD.eigenvalues()[1]), 2 * Eni * real(tmpVD.eigenvalues()[2])};
	double alphafi[3] = {-2 * Eni * imag(tmpVD.eigenvalues()[0]), -2 * Eni * imag(tmpVD.eigenvalues()[1]), -2 * Eni * imag(tmpVD.eigenvalues()[2])};

	/* Matriz de autovectores iniciales */
	Eigen::MatrixXcd Umati(3, 3);
	Eigen::MatrixXcd Umatinvi(3, 3);
	Umati = tmpVD.eigenvectors();
	Umatinvi = Umati.inverse();

	/* Matrices Cmat inicial */
	Eigen::MatrixXcd Cmati(3, 3);
	Cmati = (Umati.transpose()) * (Utemp.conjugate());

	/* Calculamos Autovalores y Autovectores finales */
	_U = make_umns(_tsf, _th, _d);
	Utemp << _U[0][0], _U[0][1], _U[0][2],
		_U[1][0], _U[1][1], _U[1][2],
		_U[2][0], _U[2][1], _U[2][2];
	Htemp << std::complex<double>(0.5 * _mss[0] / _enf, -0.5 * _alpha[0] / _enf), ProbConst::Z0, ProbConst::Z0,
		ProbConst::Z0, std::complex<double>(0.5 * _mss[1] / _enf, -0.5 * _alpha[1] / _enf), ProbConst::Z0,
		ProbConst::Z0, ProbConst::Z0, std::complex<double>(0.5 * _mss[2] / _enf, -0.5 * _alpha[2] / _enf);
	/* Hamiltoniano */
	Htemp = Utemp * Htemp * Utemp.adjoint() + _tsf * _Pot;
	tmpVD.compute(Htemp);

	/* Masas y alpha tíldes finales */
	double massffsq[3] = {2 * _enf * real(tmpVD.eigenvalues()[0]), 2 * _enf * real(tmpVD.eigenvalues()[1]), 2 * _enf * real(tmpVD.eigenvalues()[2])};
	double alphaff[3] = {-2 * _enf * imag(tmpVD.eigenvalues()[0]), -2 * _enf * imag(tmpVD.eigenvalues()[1]), -2 * _enf * imag(tmpVD.eigenvalues()[2])};

	/* Matriz de autovectores finales */
	Eigen::MatrixXcd Umatf(3, 3);
	Umatf = tmpVD.eigenvectors();

	/* Matrices Cmat final */
	Eigen::MatrixXcd Cmatinvf(3, 3);
	Cmatinvf = (Umatf.transpose()) * (Utemp.conjugate());
	Cmatinvf = Cmatinvf.inverse();

	/* Funciones de cambio de quiralidad */
	double Theta = 0;
	if (_tsi * _tsf > 0)
	{
		Theta = dGdE(_mss[_tpar], _mss[_thij], Eni, _enf, _tqcoup);
	}
	else
	{
		Theta = dGbdE(_mss[_tpar], _mss[_thij], Eni, _enf, _tqcoup);
	}

	/* Suma total */
	double sum = 0, tmp = 0, txxx = 0, txxx2 = 0;
	int i, p, h, n;

	for (i = 0; i < 3; i++){
		for (p = 0; p < 3; p++){
			for (h = 0; h < 3; h++){
				for (n = 0; n < 3; n++){
					sum = sum + real(Umatinvi.col(_tfi)[i] * conj(Umatinvi.col(_tfi)[p]) * Umatf.col(h)[_tff] * conj(Umatf.col(n)[_tff]) * (((ef / _ei) * (alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n]) - ProbConst::I * ((ef / _ei) * ((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n])))) / (square((ef / _ei) * (alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n])) + square((ef / _ei) * ((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n]))))) * (exp(-ProbConst::I * ((massffsq[h]) - (massffsq[n])) / (2 * _enf) * dist) * exp(-((alphaff[h] + alphaff[n]) / (2 * _enf)) * dist) - exp(-ProbConst::I * ((massfisq[i]) - (massfisq[p])) / (2 * Eni) * dist) * exp(-((alphafi[i] + alphafi[p]) / (2 * Eni)) * dist)) * Cmatinvf.col(h)[_thij] * conj(Cmatinvf.col(n)[_thij]) * Cmati.col(_tpar)[i] * conj(Cmati.col(_tpar)[p]));
				}
			}
		}
	}
	tmp = 2 * sum * (((ef / _ei) * _alpha[_tpar]) / Eni) * Theta;
	return tmp;
}