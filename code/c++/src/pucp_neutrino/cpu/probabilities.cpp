#include "probabilities.h"
#include "utils.h"
#include <eigen3/Eigen/Eigenvalues>
#include "integrate/_1D/GaussianQuadratures/GaussLegendre.hpp"
#include <complex>
#include <iostream>

void StandardOscilation(
	std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double** _P) {
	InvisibleDecay(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P);
}

void InvisibleDecay(
    std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double** _P) {
	Eigen::MatrixXcd Pot(3, 3);
	Eigen::MatrixXcd Hff(3, 3);
	Eigen::MatrixXcd S(3, 3);
	Eigen::MatrixXcd V(3, 3);
	double energy = _energy * 1e9;
	printf("Energy:%e\n", energy);
	printf("U:\n");
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			printf("%e + %e\t",_U[i][j].real(),_U[i][j].imag());
		}
		printf("\n");
	}
	double rho = _sigN * _rho;
    std::complex<double> DM[3][3] = { std::complex<double>(0, 0) };
	/* Matriz de masas y Decay */
	DM[0][0] = std::complex<double>(0, -0.5 * _alpha[0] / energy);
	DM[1][1] = std::complex<double>(0.5 * _dm[0] / energy, -0.5 * _alpha[1] / energy);
	DM[2][2] = std::complex<double>(0.5 * _dm[1] / energy, -0.5 * _alpha[2] / energy);
	printf("DM:\n");
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			printf("%e + %e\t", DM[i][j].real(), DM[i][j].imag());
		}
		printf("\n");
	}
	Pot << std::complex<double>(rho * 7.63247 * 0.5 * 1.e-14, 0), DM[0][1], DM[0][2],
		DM[1][0], DM[0][0], DM[1][2],
		DM[2][0], DM[2][1], DM[0][0];

	printf("Pot:\n");
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			printf("%e  + %e\t", Pot(i,j).real(), Pot(i,j).imag());
		}
		printf("\n");
	}
	
	/* Inicializando las matrices para Eigen */
	Eigen::MatrixXcd UPMNS(3, 3);
	UPMNS << _U[0][0], _U[0][1], _U[0][2],
		_U[1][0], _U[1][1], _U[1][2],
		_U[2][0], _U[2][1], _U[2][2];
	Eigen::MatrixXcd Hd(3, 3);
	Hd << DM[0][0], DM[0][1], DM[0][2],
		DM[1][0], DM[1][1], DM[1][2],
		DM[2][0], DM[2][1], DM[2][2];
	/* Hamiltoniano final efectivo */
	Hff = UPMNS * Hd;
	printf("Hff:\n");
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			printf("%e + %e\t", Hff(i,j).real(), Hff(i,j).imag());
		}
		printf("\n");
	}
	//* UPMNS.adjoint() + Pot;
	/* Calculando los autovalores y autovectores */
	Hff = Hff * UPMNS.adjoint();
	printf("Hff2:\n");
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			printf("%e + %e\t", Hff(i,j).real(), Hff(i,j).imag());
		}
		printf("\n");
	}
	Hff = Hff + Pot;
	printf("Hff3:\n");
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			printf("%e + %e\t", Hff(i,j).real(), Hff(i,j).imag());
		}
		printf("\n");
	}
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> tmp;
	tmp.compute(Hff);
	/* Calculamos la matriz S y ordenamos los autovalores */
	std::cout << "Eigen values:" << std::endl;
	std::cout << tmp.eigenvalues()[0] << "\t" << tmp.eigenvalues()[1] << "\t" << tmp.eigenvalues()[2] <<std::endl;
	V = tmp.eigenvectors();
	S << exp(-ProbConst::I * tmp.eigenvalues()[0] * _L * 1.e9 / ProbConst::GevkmToevsq), DM[0][0], DM[0][0],
		DM[0][0], exp(-ProbConst::I * tmp.eigenvalues()[1] * _L * 1.e9 / ProbConst::GevkmToevsq), DM[0][0],
		DM[0][0], DM[0][0], exp(-ProbConst::I * tmp.eigenvalues()[2] * _L * 1.e9 / ProbConst::GevkmToevsq);
	std::cout << "S:" <<   S << std::endl;
	S = (V)*S * (V.inverse());
	std::cout << "S2:" << S << std::endl;
	// Calculando la matriz de probabilidad
	std::cout << "P:" << std::endl;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			_P[i][j] = abs(S.col(i)[j] * S.col(i)[j]);
			std::cout << _P[i][j] << "\t";
		}
		std::cout << std::endl;
	} // Fila j , columna i
	
}

void NonStandardInteraction(
    std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho, 
	double* _dm, double* _parmNSI, double** _P) {
    Eigen::MatrixXcd Pot(3, 3);
    Eigen::MatrixXcd MNSI(3, 3);
    Eigen::MatrixXcd Hff(3, 3);
    Eigen::MatrixXcd S(3, 3);
    Eigen::MatrixXcd V(3, 3);
	double energy = _energy*1e9;
	printf("Energy:%e\n", energy);
	printf("U:\n");
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			printf("%e + %e\t",_U[i][j].real(),_U[i][j].imag());
		}
		printf("\n");
	}	
	double rho = _sigN*_rho;
    std::complex<double> DM[3][3] = { std::complex<double>(0, 0) };
    std::complex<double> NSI[3][3];
    DM[1][1] = std::complex<double>(0.5 * _dm[0]/energy, 0);
    DM[2][2] = std::complex<double>(0.5 * _dm[1]/energy, 0);
	printf("DM:\n");
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			printf("%e + %e\t", DM[i][j].real(), DM[i][j].imag());
		}
		printf("\n");
	}
    /* Matriz de parámetros NSI */
    NSI[0][0] = _parmNSI[0];
    NSI[0][1] = _parmNSI[3] * exp(ProbConst::I*_parmNSI[4]);
    NSI[0][2] = _parmNSI[5] * exp(ProbConst::I*_parmNSI[6]);
    NSI[1][0] = _parmNSI[3] * exp(-ProbConst::I*_parmNSI[4]);
    NSI[1][1] = _parmNSI[1];
    NSI[1][2] = _parmNSI[7] * exp(ProbConst::I*_parmNSI[8]);
    NSI[2][0] = _parmNSI[5] * exp(-ProbConst::I*_parmNSI[6]);
    NSI[2][1] = _parmNSI[7] * exp(-ProbConst::I*_parmNSI[8]);
    NSI[2][2] = _parmNSI[2];
	Pot <<
		std::complex<double>(rho*7.63247*0.5*1.e-14, 0), DM[0][1], DM[0][2],
		DM[1][0], DM[0][0], DM[1][2],
		DM[2][0], DM[2][1], DM[0][0];
	std::cout <<"Pot:" << std::endl <<  Pot << std::endl ;
	MNSI <<
		NSI[0][0], NSI[0][1], NSI[0][2],
		NSI[1][0], NSI[1][1], NSI[1][2],
		NSI[2][0], NSI[2][1], NSI[2][2];
	std::cout <<"MNSI:" << std::endl <<  MNSI << std::endl ;

	/* Inicializando las matrices para Eigen */
	Eigen::MatrixXcd UPMNS(3, 3);
	UPMNS << _U[0][0], _U[0][1], _U[0][2],
		_U[1][0], _U[1][1], _U[1][2],
		_U[2][0], _U[2][1], _U[2][2];
	Eigen::MatrixXcd Hd(3, 3);
	Hd << DM[0][0], DM[0][1], DM[0][2],
		DM[1][0], DM[1][1], DM[1][2],
		DM[2][0], DM[2][1], DM[2][2];
	// std::cout <<"Hd:" << std::endl <<  Hd << std::endl ;
	/* Hamiltoniano final efectivo */
	Hff = UPMNS * Hd * UPMNS.adjoint() + rho*7.63247*0.5*1.e-14*MNSI + Pot ;
	std::cout <<"Hff3:" << std::endl <<  Hff << std::endl ;

	/* Calculando los autovalores y autovectores */
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> tmp;
	tmp.compute(Hff);
	/* Calculamos la matriz S y ordenamos los autovalores */
	V = tmp.eigenvectors() ;
	std::cout << tmp.eigenvalues()[0] << "\t" << tmp.eigenvalues()[1] << "\t" << tmp.eigenvalues()[2] <<std::endl;
	
	S <<
		exp(-ProbConst::I*tmp.eigenvalues()[0] * _L * 1.e9/ProbConst::GevkmToevsq), DM[0][0], DM[0][0],
		DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[1] * _L * 1.e9/ProbConst::GevkmToevsq), DM[0][0],
		DM[0][0], DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[2] * _L * 1.e9/ProbConst::GevkmToevsq);
	S = ( V ) * S * ( V.inverse() );
	// Calculando la matriz de probabilidad
	std::cout << "P:" << std::endl;
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++) {
			_P[i][j] = abs(S.col(i)[j]*S.col(i)[j]);
			std::cout << _P[i][j] << "\t";
		}
		std::cout << std::endl;
	} // Fila j , columna i
}

void ViolationEquivalencePrinciple(
	std::complex<double>** _U, double _energy, int _sigN, double _L, double _rho, 
	double* _dm, double* _gamma, double** _P) {
	Eigen::MatrixXcd Pot(3, 3);
	Eigen::MatrixXcd Hff(3, 3);
	Eigen::MatrixXcd S(3, 3);
	Eigen::MatrixXcd V(3, 3);
	double energy = _energy*1e9;
	double rho = _sigN*_rho;
    std::complex<double> DM[3][3] = { std::complex<double>(0, 0) };
	/* Matriz de masas y VEP con U_g = U */
	DM[0][0] = std::complex<double>(0 + 2*energy*_gamma[0], 0);
	DM[1][1] = std::complex<double>(0.5*_dm[0]/energy + 2*energy*_gamma[1], 0);
	DM[2][2] = std::complex<double>(0.5*_dm[1]/energy + 2*energy*_gamma[2], 0);
	printf("DM:\n");
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			printf("%e + %e\t", DM[i][j].real(), DM[i][j].imag());
		}
		printf("\n");
	}
	Pot <<
		std::complex<double>(rho*7.63247*0.5*1.e-14, 0), DM[0][1], DM[0][2],
		DM[1][0], DM[0][0], DM[1][2],
		DM[2][0], DM[2][1], DM[0][0];
	std::cout <<"Pot:" << std::endl <<  Pot << std::endl ;

	/* Inicializando las matrices para Eigen */
	Eigen::MatrixXcd UPMNS(3, 3);
	UPMNS << _U[0][0], _U[0][1], _U[0][2],
		_U[1][0], _U[1][1], _U[1][2],
		_U[2][0], _U[2][1], _U[2][2];
	Eigen::MatrixXcd Hd(3, 3);
	Hd << DM[0][0], DM[0][1], DM[0][2],
		DM[1][0], DM[1][1], DM[1][2],
		DM[2][0], DM[2][1], DM[2][2];
	/* Hamiltoniano final efectivo */
	Hff = UPMNS * Hd * UPMNS.adjoint() + Pot ;
	std::cout <<"Hff3:" << std::endl <<  Hff << std::endl ;
	/* Calculando los autovalores y autovectores */
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> tmp;
	tmp.compute(Hff);
	/* Calculamos la matriz S y ordenamos los autovalores */
	V = tmp.eigenvectors() ;
	std::cout << tmp.eigenvalues()[0] << "\t" << tmp.eigenvalues()[1] << "\t" << tmp.eigenvalues()[2] <<std::endl;

	S <<
		exp(-ProbConst::I*tmp.eigenvalues()[0]*_L*1.e9/ProbConst::GevkmToevsq), DM[0][0], DM[0][0],
		DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[1]*_L*1.e9/ProbConst::GevkmToevsq), DM[0][0],
		DM[0][0], DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[2]*_L*1.e9/ProbConst::GevkmToevsq);
	S = ( V ) * S * ( V.inverse() );
	// Calculando la matriz de probabilidad
	std::cout << "P:" << std::endl;
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) {
			_P[i][j] = abs(S.col(i)[j]*S.col(i)[j]);
			std::cout << _P[i][j] << "\t";

		}
		std::cout << std::endl;
	}
}

void VisibleDecay(double _energy, double _L, double _rho, double* _th,
	double* _dm, double d,  double* _alpha, double _mlight, int _tfi, int _tsi, int _tff, 
	int _tsf, int _tpar, int _thij, int _tqcoup, double* _P) {
	// Matriz Potencial
	Eigen::Matrix3cd Pot; //(3, 3)
	/* Algoritmo para Integral */
	_1D::GQ::GaussLegendreQuadrature<double, 64> NintegrGLQ; // 64
	double Enf = _energy * 1.e9;
	double* mss = new double[3];
	double mm;	/* variable temporal */
	/* definimos vector mss */
	if (_dm[1] > 0) {
		mss[0] = _mlight;
		mss[1] = _dm[0] + _mlight;
		mss[2] = _dm[1] + _mlight;
	} else {
		mss[0] = _mlight;
		mss[1] = -_dm[1] + _mlight;
		mss[2] = _dm[0] - _dm[1] + _mlight;
	}
	mm = min(20, (mss[_tpar] / mss[_thij]) * _energy); // Límite superior de integración
	// Inicializamos el potencial
	Pot << std::complex<double>(_rho * 7.63247 * 0.5 * 1.e-14, 0), ProbConst::Z0, ProbConst::Z0,
		ProbConst::Z0, ProbConst::Z0, ProbConst::Z0, 
		ProbConst::Z0, ProbConst::Z0, ProbConst::Z0;
	using std::placeholders::_1;
	*_P = 1.e9 * NintegrGLQ(std::bind(
		IntegrandoPhi, _1, mss, Pot, Enf, _L, _th, d, _alpha, _tfi, _tsi, _tff, 
		_tsf, _tpar, _thij, _tqcoup), (1.00001) * _energy, mm);
}