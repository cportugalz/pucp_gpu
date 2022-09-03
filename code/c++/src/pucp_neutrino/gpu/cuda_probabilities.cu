#include "cuda_probabilities.h"
#include <cuda_runtime.h>
#include <time.h>
using namespace std;


// // GPU Standard Oscilation
// void cuda_StandardOscilation(
// 	cuDoubleComplex* _U, double* _energy, int _size_data, int _sigN, double _L, double _rho,
// 	double* _dm, double* _alpha, double* _events){
// 		cuda_InvisibleDecay(_U, _energy, _size_data, _sigN, _L, _rho, _dm, _alpha, _events);
// }



const std::complex<double> ProbConst::I = std::complex<double>(0, 1);
const std::complex<double> ProbConst::Z0 = std::complex<double>(0, 0);
const double ProbConst::hbar = 6.58211928*1.e-25;
const double ProbConst::clight = 299792458;
//double GevkmToevsq = hbar*clight*1.e15;
const double ProbConst::GevkmToevsq = 0.197327; // Approximated value


// Working on host code the matrix U but returning it as column form data_type
cuDoubleComplex* make_umns(int _sg, double* _th, double _dcp) {
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
	cuDoubleComplex* cuU = new cuDoubleComplex[9];
	// printf("U:\n");
	for(int i=0; i<3; i++){
		for(int j=0; j<3;j++){
			// printf("%f %f\t", U[i][j].real(), U[i][j].imag());
			cuU[i+j*3] = make_cuDoubleComplex(U[i][j].real(), U[i][j].imag());
			// printf("U[%d][%d]:(%e %e)\t ", i,j,cuU[i+j*3].x, cuU[i+j*3].y);
		}
		// printf("\n");
	}
	for (int i=0; i<3; i++){
		delete[] U[i];
	}
	delete[] U;
	return cuU;
}

