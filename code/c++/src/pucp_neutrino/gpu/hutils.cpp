#include "hutils.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			std::printf("%e ", A[j * lda + i]);
		}
		std::printf("\n");
	}
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			std::printf("%e ", A[j * lda + i]);
		}
		std::printf("\n");
	}
}

template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			std::printf("%e + %ej ", A[j * lda + i].x, A[j * lda + i].y);
		}
		std::printf("\n");
	}
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			std::printf("%e + %ej ", A[j * lda + i].x, A[j * lda + i].y);
		}
		std::printf("\n");
	}
}



template <> void print_vector(const int &m, const float *A) {
	for (int i = 0; i < m; i++) {
		std::printf("%0.2f ", A[i]);
	}
	std::printf("\n");
}

template <> void print_vector(const int &m, const double *A) {
	for (int i = 0; i < m; i++) {
		std::printf("%0.2f ", A[i]);
	}
	std::printf("\n");
}

template <> void print_vector(const int &m, const cuComplex *A) {
	for (int i = 0; i < m; i++) {
		std::printf("%0.2f + %0.2fj ", A[i].x, A[i].y);
	}
	std::printf("\n");
}

template <> void print_vector(const int &m, const cuDoubleComplex *A) {
	for (int i = 0; i < m; i++) {
		std::printf("%0.2f + %0.2fj ", A[i].x, A[i].y);
	}
	std::printf("\n");
}



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
