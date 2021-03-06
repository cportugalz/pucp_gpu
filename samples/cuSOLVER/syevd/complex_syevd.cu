#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;

	const int m = 3;
	const int lda = m;
	/*
	 *       | 3.5 0.5 0.0 |
	 *   A = | 0.5 3.5 0.0 |
	 *       | 0.0 0.0 2.0 |
	 *
	 */
	const std::vector<double> A = {
		make_cuDoubleComplex(2.068427e-12, 0.000000e+00),
		make_cuDoubleComplex(5.698705e-13, -6.813775e-12),
		make_cuDoubleComplex(-6.248490e-13, -6.107545e-12 ),
		make_cuDoubleComplex(5.698705e-13, 6.813775e-12),
		make_cuDoubleComplex(3.445797e-11, 1.262177e-29),
		make_cuDoubleComplex( 2.974163e-11,1.278085e-13),
		make_cuDoubleComplex(-6.248490e-13, 6.107545e-12),
		make_cuDoubleComplex(2.974163e-11, -1.278085e-13),
		make_cuDoubleComplex(2.793644e-11, 0.000000e+00)
	};
	const std::vector<double> lambda = {2.0, 3.0, 4.0};

	std::vector<double> V(lda * m, 0); // eigenvectors
	std::vector<double> W(m, 0);       // eigenvalues

	double *d_A = nullptr;
	double *d_W = nullptr;
	int *d_info = nullptr;

	int info = 0;

	int lwork = 0;            /* size of workspace */
	double *d_work = nullptr; /* device workspace*/

	std::printf("A = (matlab base-1)\n");
	print_matrix(m, m, A.data(), lda);
	std::printf("=====\n");

	/* step 1: create cusolver handle, bind a stream */
	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * W.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

	CUDA_CHECK(
		cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

	// step 3: query working space of syevd
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

	// step 4: compute spectrum
	CUSOLVER_CHECK(
		cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, d_info));

	CUDA_CHECK(
		cudaMemcpyAsync(V.data(), d_A, sizeof(double) * V.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(
		cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	std::printf("after syevd: info = %d\n", info);
	if (0 > info) {
		std::printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}

	std::printf("eigenvalue = (matlab base-1), ascending order\n");
	int idx = 1;
	for (auto const &i : W) {
		std::printf("W[%i] = %E\n", idx, i);
		idx++;
	}

	std::printf("V = (matlab base-1)\n");
	print_matrix(m, m, V.data(), lda);
	std::printf("=====\n");

	// step 4: check eigenvalues
	double lambda_sup = 0;
	for (int i = 0; i < m; i++) {
		double error = fabs(lambda[i] - W[i]);
		lambda_sup = (lambda_sup > error) ? lambda_sup : error;
	}
	std::printf("|lambda - W| = %E\n", lambda_sup);

	/* free resources */
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_W));
	CUDA_CHECK(cudaFree(d_info));
	CUDA_CHECK(cudaFree(d_work));

	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

	CUDA_CHECK(cudaStreamDestroy(stream));

	CUDA_CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}
