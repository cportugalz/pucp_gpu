#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;
	syevjInfo_t syevj_params = NULL;

	const int m = 3;
	const int lda = m;
	const int batchSize = 2;
	/*
	 *        |  1  -1   0 |
	 *   A0 = | -1   2   0 |
	 *        |  0   0   0 |
	 *
	 *   A0 = V0 * W0 * V0**T
	 *
	 *   W0 = diag(0, 0.3820, 2.6180)
	 *
	 *        |  3   4  0 |
	 *   A1 = |  4   7  0 |
	 *        |  0   0  0 |
	 *
	 *   A1 = V1 * W1 * V1**T
	 *
	 *   W1 = diag(0, 0.5279, 9.4721)
	 *
	 */

	std::vector<cuDoubleComplex> A(lda * m * batchSize, {0,0}); /* V = [A0 ; A1] */
	std::vector<cuDoubleComplex> V(lda * m * batchSize, {0,0}); /* V = [V0 ; V1] */
	std::vector<double> W(m * batchSize, 0);       /* W = [W0 ; W1] */
	std::vector<int> info(batchSize, 0);           /* info = [info0 ; info1] */

	cuDoubleComplex *d_A = nullptr;    /* lda-by-m-by-batchSize */
	double *d_W = nullptr;    /* m-by-batchSize */
	int *d_info = nullptr;    /* batchSize */
	cuDoubleComplex *d_work = nullptr; /* device workspace for syevjBatched */
	int lwork = 0;            /* size of workspace */

	/* configuration of syevj  */
	const double tol = 1.e-9;
	const int max_sweeps = 100;
	const int sort_eig = 1;                                  /* don't sort eigenvalues */
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute eigenvectors */
	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	cuDoubleComplex *A0 = A.data();
	cuDoubleComplex *A1 = A.data() + lda * m;
	/*
	 *        |  1  -1   0 |
	 *   A0 = | -1   2   0 |
	 *        |  0   0   0 |
	 *   A0 is column-major
	 */
			 
		 
	   
	A0[0 + 0 * lda] = make_cuDoubleComplex(4.024018e-12, 0);
	A0[1 + 0 * lda] = make_cuDoubleComplex(1.139741e-12,-1.362755e-11);
	A0[2 + 0 * lda] = make_cuDoubleComplex(-1.249698e-12,-1.221509e-11);
	A0[0 + 1 * lda] = make_cuDoubleComplex(1.139741e-12, 1.362755e-11 );
	A0[1 + 1 * lda] = make_cuDoubleComplex(6.891594e-11, 2.524355e-29);
	A0[2 + 1 * lda] = make_cuDoubleComplex(5.948327e-11, 2.556170e-13);
	A0[0 + 2 * lda] = make_cuDoubleComplex(-1.249698e-12, 1.221509e-11);
	A0[1 + 2 * lda] = make_cuDoubleComplex(5.948327e-11, -2.556170e-13);
	A0[2 + 2 * lda] = make_cuDoubleComplex(5.587288e-11, 0.000000e+00);
	// A0[0 + 0 * lda] = make_cuDoubleComplex(2,0);
	// A0[1 + 0 * lda] = make_cuDoubleComplex(0,1);
	// A0[2 + 0 * lda] = make_cuDoubleComplex(0,0);
	// A0[0 + 1 * lda] = make_cuDoubleComplex(0,1);
	// A0[1 + 1 * lda] = make_cuDoubleComplex(1,0);
	// A0[2 + 1 * lda] = make_cuDoubleComplex(0,5);
	// A0[0 + 2 * lda] = make_cuDoubleComplex(0,0);
	// A0[1 + 2 * lda] = make_cuDoubleComplex(0,5);
	// A0[2 + 2 * lda] = make_cuDoubleComplex(4,0);
	/*
	 *        |  3   4  0 |
	 *   A1 = |  4   7  0 |
	 *        |  0   0  0 |
	 *   A1 is column-major
	 */
			   
	A1[0 + 0 * lda] = make_cuDoubleComplex(2.068427e-12, 0.000000e+00);
	A1[1 + 0 * lda] = make_cuDoubleComplex(5.698705e-13, -6.813775e-12);
	A1[2 + 0 * lda] = make_cuDoubleComplex(-6.248490e-13, -6.107545e-12 );
	A1[0 + 1 * lda] = make_cuDoubleComplex(5.698705e-13, 6.813775e-12);
	A1[1 + 1 * lda] = make_cuDoubleComplex(3.445797e-11, 1.262177e-29);
	A1[2 + 1 * lda] = make_cuDoubleComplex( 2.974163e-11,1.278085e-13);
	A1[0 + 2 * lda] = make_cuDoubleComplex(-6.248490e-13, 6.107545e-12);
	A1[1 + 2 * lda] = make_cuDoubleComplex(2.974163e-11, -1.278085e-13);
	A1[2 + 2 * lda] = make_cuDoubleComplex(2.793644e-11, 0.000000e+00);
	// A1[0 + 0 * lda] = make_cuDoubleComplex(0,2);
	// A1[1 + 0 * lda] = make_cuDoubleComplex(0,-1);
	// A1[2 + 0 * lda] = make_cuDoubleComplex(0,0);
	// A1[0 + 1 * lda] = make_cuDoubleComplex(0,1);
	// A1[1 + 1 * lda] = make_cuDoubleComplex(1,0);
	// A1[2 + 1 * lda] = make_cuDoubleComplex(0,-5);
	// A1[0 + 2 * lda] = make_cuDoubleComplex(0,0);
	// A1[1 + 2 * lda] = make_cuDoubleComplex(0,5);
	// A1[2 + 2 * lda] = make_cuDoubleComplex(0,4);

	std::printf("A0 = (matlab base-1)\n");
	print_matrix(m, m, A0, lda);
	std::printf("=====\n");

	std::printf("A1 = (matlab base-1)\n");
	print_matrix(m, m, A1, lda);
	std::printf("=====\n");

	/* step 1: create cusolver handle, bind a stream */
	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

	/* step 2: configuration of syevj */
	CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));

	/* default value of tolerance is machine zero */
	CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));

	/* default value of max. sweeps is 100 */
	CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));

	/* disable sorting */
	CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));

	/* step 3: copy A to device */
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * W.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * info.size()));

	CUDA_CHECK(
		cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
	/* step 4: query working space of syevj */
	CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W,
													  &lwork, syevj_params, batchSize));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(cuDoubleComplex) * lwork));

	/* step 5: compute eigen-pair   */
	CUSOLVER_CHECK(cusolverDnZheevjBatched(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork,
										   d_info, syevj_params, batchSize));

	CUDA_CHECK(
		cudaMemcpyAsync(V.data(), d_A, sizeof(double) * A.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(
		cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(),
							   cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	for (int i = 0; i < batchSize; i++) {
		if (0 == info[i]) {
			std::printf("matrix %d: syevj converges \n", i);
		} else if (0 > info[i]) {
			/* only info[0] shows if some input parameter is wrong.
			 * If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
			 */
			std::printf("Error: %d-th parameter is wrong \n", -info[i]);
			exit(1);
		} else { /* info = m+1 */
				 /* if info[i] is not zero, Jacobi method does not converge at i-th matrix. */
			std::printf("WARNING: matrix %d, info = %d : sygvj does not converge \n", i, info[i]);
		}
	}

	/* Step 6: show eigenvalues and eigenvectors */
	double *W0 = W.data();
	double *W1 = W.data() + m;

	std::printf("==== \n");
	for (int i = 0; i < m; i++) {
		std::printf("W0[%d] = %e \n", i, W0[i]);
	}
	std::printf("==== \n");
	for (int i = 0; i < m; i++) {
		std::printf("W1[%d] = %e \n", i, W1[i]);
	}
	std::printf("==== \n");

	cuDoubleComplex *V0 = V.data();
	cuDoubleComplex *V1 = V.data() + lda * m;

	std::printf("V0 = (matlab base-1)\n");
	print_matrix(m, m, V0, lda);
	std::printf("V1 = (matlab base-1)\n");
	print_matrix(m, m, V1, lda);

	/* free resources */
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_W));
	CUDA_CHECK(cudaFree(d_info));
	CUDA_CHECK(cudaFree(d_work));

	CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));

	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

	CUDA_CHECK(cudaStreamDestroy(stream));

	CUDA_CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}
