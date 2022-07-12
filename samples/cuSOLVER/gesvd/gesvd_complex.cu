#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;

	const int m = 3;   /* 1 <= m <= 32 */
	const int n = 3;   /* 1 <= n <= 32 */
	const int lda = m; /* lda >= m */


	// const std::vector<cuDoubleComplex> A = {
	// 	make_cuDoubleComplex(4.024018e-12, 0),
	// 	make_cuDoubleComplex(1.139741e-12,-1.362755e-11),
	// 	make_cuDoubleComplex(-1.249698e-12,-1.221509e-11),
	// 	make_cuDoubleComplex(1.139741e-12, 1.362755e-11 ),
	// 	make_cuDoubleComplex(6.891594e-11, 2.524355e-29),
	// 	make_cuDoubleComplex(5.948327e-11, 2.556170e-13),
	// 	make_cuDoubleComplex(-1.249698e-12, 1.221509e-11),
	// 	make_cuDoubleComplex(5.948327e-11, -2.556170e-13),
	// 	make_cuDoubleComplex(5.587288e-11, 0.000000e+00)
	// };
	const std::vector<cuDoubleComplex> A = {
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
	std::vector<cuDoubleComplex> U(lda * m, {0,0});  /* m-by-m unitary matrix, left singular vectors  */
	std::vector<cuDoubleComplex> VT(lda * n, {0,0}); /* n-by-n unitary matrix, right singular vectors */
	std::vector<double> S(n, 0.0);        /* numerical singular value */
	std::vector<double> S_exact = {7.065283497082729,
								   1.040081297712078}; /* exact singular values */
	int info_gpu = 0;                                  /* host copy of error info */

	cuDoubleComplex *d_A = nullptr;
	double *d_S = nullptr;  /* singular values */
	cuDoubleComplex *d_U = nullptr;  /* left singular vectors */
	cuDoubleComplex *d_VT = nullptr; /* right singular vectors */
	cuDoubleComplex *d_W = nullptr;  /* W = S*VT */

	int *devInfo = nullptr;

	int lwork = 0; /* size of workspace */
	cuDoubleComplex *d_work = nullptr;
	double *d_rwork = nullptr;

	const double h_one = 1;
	const double h_minus_one = -1;

	std::printf("A = (matlab base-1)\n");
	print_matrix(m, n, A.data(), lda);
	std::printf("=====\n");

	/* step 1: create cusolver handle, bind a stream */
	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
	CUBLAS_CHECK(cublasCreate(&cublasH));

	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

	/* step 2: copy A to device */
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cuDoubleComplex) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(cuDoubleComplex) * U.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(cuDoubleComplex) * VT.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(cuDoubleComplex) * lda * n));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

	CUDA_CHECK(
		cudaMemcpyAsync(d_A, A.data(), sizeof(cuDoubleComplex) * A.size(), cudaMemcpyHostToDevice, stream));

	/* step 3: query working space of SVD */
	CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(cusolverH, m, n, &lwork));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(cuDoubleComplex) * lwork));

	/* step 4: compute SVD*/
	signed char jobu = 'N';  // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	CUSOLVER_CHECK(cusolverDnZgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda, d_S, d_U,
									lda, // ldu
									d_VT,
									lda, // ldvt,
									d_work, lwork, d_rwork, devInfo));

	CUDA_CHECK(
		cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(cuDoubleComplex) * VT.size(), cudaMemcpyDeviceToHost,
							   stream));
	CUDA_CHECK(
		cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	std::printf("after gesvd: info_gpu = %d\n", info_gpu);
	if (0 == info_gpu) {
		std::printf("gesvd converges \n");
	} else if (0 > info_gpu) {
		std::printf("%d-th parameter is wrong \n", -info_gpu);
		exit(1);
	} else {
		std::printf("WARNING: info = %d : gesvd does not converge \n", info_gpu);
	}

	std::printf("S = singular values (matlab base-1)\n");
	print_matrix(n, 1, S.data(), n);
	std::printf("=====\n");

	std::printf("U = left singular vectors (matlab base-1)\n");
	print_matrix(m, m, U.data(), lda);
	std::printf("=====\n");

	std::printf("VT = right singular vectors (matlab base-1)\n");
	print_matrix(n, n, VT.data(), lda);
	std::printf("=====\n");

	// step 5: measure error of singular value
	// double ds_sup = 0;
	// for (int j = 0; j < n; j++) {
	// 	double err = fabs(S[j] - S_exact[j]);
	// 	ds_sup = (ds_sup > err) ? ds_sup : err;
	// }
	// std::printf("|S - S_exact| = %E \n", ds_sup);

	// CUBLAS_CHECK(cublasZdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n, d_VT, lda, d_S, 1, d_W, lda));

	// CUDA_CHECK(
	// 	cudaMemcpyAsync(d_A, A.data(), sizeof(double) * lda * n, cudaMemcpyHostToDevice, stream));

	// CUBLAS_CHECK(cublasDgemm(cublasH,
	// 						 CUBLAS_OP_N,  // U
	// 						 CUBLAS_OP_N,  // W
	// 						 m,            // number of rows of A
	// 						 n,            // number of columns of A
	// 						 n,            // number of columns of U
	// 						 &h_minus_one, /* host pointer */
	// 						 d_U,          // U
	// 						 lda,
	// 						 d_W,         // W
	// 						 lda, &h_one, /* hostpointer */
	// 						 d_A, lda));

	// double dR_fro = 0.0;
	// CUBLAS_CHECK(cublasDnrm2(cublasH, lda * n, d_A, 1, &dR_fro));

	// std::printf("|A - U*S*VT| = %E \n", dR_fro);

	/* free resources */
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_U));
	CUDA_CHECK(cudaFree(d_VT));
	CUDA_CHECK(cudaFree(d_S));
	CUDA_CHECK(cudaFree(d_W));
	CUDA_CHECK(cudaFree(devInfo));
	CUDA_CHECK(cudaFree(d_work));
	CUDA_CHECK(cudaFree(d_rwork));

	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
	CUBLAS_CHECK(cublasDestroy(cublasH));

	CUDA_CHECK(cudaStreamDestroy(stream));

	CUDA_CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}
