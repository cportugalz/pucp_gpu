#include "utils.h"
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
