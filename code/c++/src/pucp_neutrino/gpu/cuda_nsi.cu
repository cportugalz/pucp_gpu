#include "cuda_probabilities.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"
#include <cusolverDn.h>
#include <time.h>
using namespace std;


// GPU Kernel for visible decay
__global__ void gpu_nsi(
	cuDoubleComplex* _U, int _size_data, int _sigN, double _L, double _rho,
	double* _dm, double* _alpha, double* _events, cuDoubleComplex** _batchedDM,
	cuDoubleComplex** _batchedPot, cuDoubleComplex** _batchedU){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < _size_data){
		//Copying _size_data of U to batchedU
		for (int i=0; i<9; i++) {
			_batchedU[tid][i] = _U[i];
		}
		double energy = (tid+1)/100.0 * 1e9;
		// printf("Calling threadIdx: %d  for energy %e\n", tid, energy);
		double rho = _sigN * _rho;
		// Matriz de masas y Decay
		_batchedDM[tid][4] = {0.5 * _dm[0] / energy, 0};
		_batchedDM[tid][8] = {0.5 * _dm[1] / energy, 0};

		_batchedPot[tid][0] = {rho * 7.63247 * 0.5 * 1.e-14, 0}; _batchedPot[tid][3] = _batchedDM[tid][3];
		_batchedPot[tid][6]= _batchedDM[tid][6]; _batchedPot[tid][1] = _batchedDM[tid][1]; _batchedPot[tid][4] = _batchedDM[tid][0];
		_batchedPot[tid][7] = _batchedDM[tid][7]; _batchedPot[tid][2] = _batchedDM[tid][2]; _batchedPot[tid][5] = _batchedDM[tid][5];
		_batchedPot[tid][8] = _batchedDM[tid][0];
	}
}


// GPU  Non Standard Interaction
void cuda_NonStandardInteraction(
	cuDoubleComplex* _U, int _batch_count, int _sigN, double _L, double _rho,
	double* _dm, double* _alpha, double* _events) {
		float threads = 1024;
		using data_type =  cuDoubleComplex;
		const int m = 3;
		const int n = 3;
		const int k = 3;
		const int lda = 3;
		const int ldb = 3;
		const int ldc = 3;
		const int ldu = m; /* ldu >= m */
		const int ldv = n; /* ldv >= n */
		const int minmn = (m < n) ? m : n; /* min(m,n) */

		int blocks = ceil(_batch_count/threads);
		printf("Assigning host memory for Non Standard Interaction.\n");

		// Assigning memory to batched matrices of mass and decay
		data_type** batchedU = nullptr;
		// data_type* host_batchedU[_batch_count];
		data_type* device_batchedU[_batch_count];

		data_type** batchedDM = nullptr;
		data_type* host_batchedDM[_batch_count];
		data_type* device_batchedDM[_batch_count];

		data_type** batchedPot = nullptr;
		data_type* host_batchedPot[_batch_count];
		data_type* device_batchedPot[_batch_count];

		// data_type** batchedHff = nullptr;
		// data_type* host_batchedHff[_batch_count];
		// data_type* device_batchedHff[_batch_count];

		// data_type** batchedHff2 = nullptr;
		// data_type* host_batchedHff2[_batch_count];
		// data_type* device_batchedHff2[_batch_count];

		// data_type** batchedHff3 = nullptr;
		// data_type* host_batchedHff3[_batch_count];
		// data_type* device_batchedHff3[_batch_count];

		// data_type** batchedS = nullptr;
		// data_type* host_batchedS[_batch_count];
		// data_type* device_batchedS[_batch_count];

		// data_type** batchedS1= nullptr;
		// data_type* host_batchedS1[_batch_count];
		// data_type* device_batchedS1[_batch_count];

		// data_type** batchedS2 = nullptr;
		// data_type* host_batchedS2[_batch_count];
		// data_type* device_batchedS2[_batch_count];

		// data_type** batchedV = nullptr;
		// data_type* host_batchedV[_batch_count];
		// data_type* device_batchedV[_batch_count];

		// data_type** batchedInvV = nullptr;
		// data_type* host_batchedInvV[_batch_count];
		// data_type* device_batchedInvV[_batch_count];

		// double** batchedP = nullptr;
		// double* host_batchedP[_batch_count];
		// double* device_batchedP[_batch_count];

		for (int i = 0; i < _batch_count; i++) {
			// cudaMallocHost(&host_batchedU[i],   sizeof(data_type)*m*n);
			cudaMallocHost(&host_batchedDM[i],  sizeof(data_type)*m*n);
			cudaMallocHost(&host_batchedPot[i], sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedHff[i], sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedHff2[i],sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedHff3[i],sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedS[i] ,  sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedS1[i],  sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedS2[i],  sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedV[i],   sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedInvV[i],sizeof(data_type)*m*n);
			// cudaMallocHost(&host_batchedP[i],   sizeof(double)*m*n);

			for (int j = 0; j < m*n; j++){
				// host_batchedU[i][j]  = make_cuDoubleComplex(0.0,0.0);
				host_batchedDM[i][j]  = make_cuDoubleComplex(0.0,0.0);
				host_batchedPot[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedHff[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedHff2[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedHff3[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedS[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedS1[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedS2[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedV[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedInvV[i][j] = make_cuDoubleComplex(0.0,0.0);
				// host_batchedP[i][j] = 0;
			}
		}
		printf("Assigning device memory for Non Standard Interaction.\n");
		// double* denergy = nullptr;
		// cudaMalloc(reinterpret_cast<void **>(&denergy), _batch_count * sizeof(double));

		for (int i = 0; i < _batch_count; i++) {
			cudaMalloc(reinterpret_cast<void **>(&device_batchedU[i]),		sizeof(data_type) * m*n);
			cudaMalloc(reinterpret_cast<void **>(&device_batchedDM[i]),		sizeof(data_type) * m*n);
			cudaMalloc(reinterpret_cast<void **>(&device_batchedPot[i]),	sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedHff[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedHff2[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedHff3[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedS[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedS1[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedS2[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedV[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedInvV[i]), sizeof(data_type) * m*n);
			// cudaMalloc(reinterpret_cast<void **>(&device_batchedP[i]), sizeof(double) * m*n));
		}

		cudaMalloc(reinterpret_cast<void **>(&batchedU),  	_batch_count*sizeof(data_type *));
		cudaMalloc(reinterpret_cast<void **>(&batchedDM),  	_batch_count*sizeof(data_type *));
		cudaMalloc(reinterpret_cast<void **>(&batchedPot), 	_batch_count*sizeof(data_type *));
		// cudaMalloc(reinterpret_cast<void **>(&batchedHff), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedHff2), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedHff3), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedS), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedS1), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedS2), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedV), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedInvV), 	_batch_count*sizeof(data_type *)));
		// cudaMalloc(reinterpret_cast<void **>(&batchedP), 	_batch_count*sizeof(double *)));
		printf("Copying host memory to device memory.\n");
		for (int i = 0; i < _batch_count; i++) {
			// cudaMemcpy(device_batchedU[i], host_batchedU[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice);
			cudaMemcpy(device_batchedDM[i], host_batchedDM[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice);
			cudaMemcpy(device_batchedPot[i], host_batchedPot[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice);
			// cudaMemcpy(device_batchedHff[i], host_batchedHff[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedHff2[i], host_batchedHff2[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedHff3[i], host_batchedHff3[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedS[i], host_batchedS[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedS1[i], host_batchedS1[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedS2[i], host_batchedS2[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedV[i], host_batchedV[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedInvV[i], host_batchedInvV[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			// cudaMemcpy(device_batchedP[i], host_batchedP[i], sizeof(double) * m*n, cudaMemcpyHostToDevice));
		}
		cudaMemcpy(batchedU, device_batchedU, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		cudaMemcpy(batchedDM, device_batchedDM, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		cudaMemcpy(batchedPot, device_batchedPot, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedHff, device_batchedHff, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedHff2, device_batchedHff2, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedHff3, device_batchedHff3, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedS, device_batchedS, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedS1, device_batchedS1, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedS2, device_batchedS2, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedV, device_batchedV, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedInvV, device_batchedInvV, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		// cudaMemcpy(batchedP, device_batchedP, sizeof(double*)*_batch_count, cudaMemcpyHostToDevice);
		// clock_t start_time = clock();
		printf("Calling NSI Kernel with %d blocks and %.0f threads per block\n", blocks, threads);
		gpu_nsi<<<blocks, threads>>>(
			_U, _batch_count, _sigN, _L, _rho, _dm, _alpha, _events,
			batchedDM, batchedPot, batchedU);
		cudaDeviceSynchronize();
		// // CuBlas Operations
		// cublasHandle_t cublasH = NULL;
		// // cudaStream_t stream = NULL;

		// const data_type alpha = {1.0, 0.0};
		// const data_type beta = {1.0, 0.0};

		// CUBLAS_CHECK(cublasCreate(&cublasH));
		// // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		// // CUBLAS_CHECK(cublasSetStream(cublasH, stream));
		// CUBLAS_CHECK(cublasZgemmBatched(
		// 	cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (const data_type * const *)batchedU, lda,
		// 	(const data_type*  const*) batchedDM, ldb, &beta, batchedHff, ldc, _batch_count));
		// // cudaStreamSynchronize(stream));
		// CUBLAS_CHECK(cublasZgemmBatched(
		// 	cublasH, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, &alpha, (const data_type * const *)batchedHff, lda,
		// 	(const data_type*  const*) batchedU, ldb, &beta, batchedHff2, ldc, _batch_count));
		// // cudaStreamSynchronize(stream));
		// // CUBLAS_CHECK(cublasDestroy(cublasH));

		// // Calculating eigen value with cuSolver
		// cusolverDnHandle_t cusolverH = NULL;
		// gesvdjInfo_t gesvdj_params = NULL;
		// data_type *d_A = nullptr;    /* lda-by-m-by-batchSize */
		// data_type *h_A = (data_type*) malloc(sizeof(data_type) * _batch_count * m * n);    /* lda-by-m-by-batchSize */
		// data_type *d_U = nullptr;    /* lda-by-m-by-batchSize */
		// data_type *h_U = (data_type*) malloc( sizeof(data_type) * ldu * m * _batch_count);    /* lda-by-m-by-_batch_count */
		// data_type *d_V = nullptr;    /* lda-by-m-by-_batch_count */
		// data_type *h_V = (data_type*) malloc( sizeof(data_type) * ldv * n * _batch_count);    /* lda-by-m-by-_batch_count */
		// double* S = (double*) malloc(sizeof(cuDoubleComplex) * minmn * _batch_count);
		// double *d_S = nullptr; /* minmn-by-batchSize */
		// int* info = (int*) malloc(sizeof(int) * _batch_count);
		// int *d_info = nullptr; /* batchSize */


		// int lwork = 0;            /* size of workspace */
		// data_type *d_work = nullptr; /* device workspace for getrf */

		// // const double tol = 1.e-7;
		// // const int max_sweeps = 50;
		// const int sort_svd = 0;                                  /* don't sort singular values */
		// const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */
		// cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * _batch_count * m * n));
		// sum_batched<<<blocks, threads>>> (batchedPot, batchedHff2, batchedHff3, d_A, _batch_count);
		// cudaDeviceSynchronize();
		// // Hff[0] = U[0] * DM[0] * UC[0] + Pot[0]
		// // Hff[1] = U[1] * DM[1] * UC[1] + Pot[1]
		// // Hff[N-1] = U[N-1] * DM[N-1] * UC[N-1] + Pot[N-1]
		// /* step 1: create cusolver handle, bind a stream */
		// cusolverDnCreate(&cusolverH);
		// // cusolverDnSetStream(cusolverH, stream);

		// /* step 2: configuration of syevj */
		// cusolverDnCreateGesvdjInfo(&gesvdj_params);

		// /* default value of tolerance is machine zero */
		// // cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);

		// /* default value of max. sweeps is 100 */
		// // cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);


		// /* disable sorting */
		// cusolverDnXgesvdjSetSortEig(gesvdj_params, sort_svd);
		// cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * ldu * m * _batch_count));
		// cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(data_type) * ldv * n * _batch_count));
		// cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * minmn * _batch_count));
		// cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * _batch_count));

		// cusolverDnZgesvdjBatched_bufferSize(cusolverH, jobz, m, n, d_A, lda, d_S, d_U,
		// 	ldu, d_V, ldv, &lwork, gesvdj_params, _batch_count);
		// cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));

		// cusolverDnZgesvdjBatched(cusolverH, jobz, m, n, d_A, lda, d_S, d_U, ldu, d_V,
		// 	ldv, d_work, lwork, d_info, gesvdj_params, _batch_count);

		// // building S from eigenvalues and ordering V to array of vectors
		// building_SandV<<<blocks, threads>>> (
		// 	batchedS, batchedDM, d_S, d_V, batchedV, _batch_count,
		// 	make_cuDoubleComplex(ProbConst::I.real(), ProbConst::I.imag()),
		// 	ProbConst::GevkmToevsq, _L);
		// // building the inverse of V from batchedV
		// CUBLAS_CHECK(
		// 	cublasZmatinvBatched(
		// 		cublasH, n, (const cuDoubleComplex * const *) batchedV, lda, batchedInvV, lda,
		// 		d_info, _batch_count)
		// );
		// // S = S*V * Vinv
		// CUBLAS_CHECK(cublasZgemmBatched(
		// 	cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (const data_type * const *)batchedV, lda,
		// 	(const data_type*  const*) batchedS, ldb, &beta, batchedS1, ldc, _batch_count));
		// CUBLAS_CHECK(cublasZgemmBatched(
		// 	cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (const data_type * const *)batchedS1, lda,
		// 	(const data_type*  const*) batchedInvV, ldb, &beta, batchedS2, ldc, _batch_count));
		// buildP<<<blocks, threads>>>(batchedP, batchedS2, _batch_count);
		// clock_t stop_time = clock();
		// printf("Computation time: %.7fs\n", (double)(stop_time - start_time)/CLOCKS_PER_SEC);
		for (int i = 0; i < _batch_count; i++) {
			// cudaMemcpy(host_batchedU[i], device_batchedU[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedDM[i], device_batchedDM[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedPot[i], device_batchedPot[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			// cudaMemcpy(host_batchedHff[i], device_batchedHff[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			// cudaMemcpy(host_batchedHff2[i], device_batchedHff2[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			// cudaMemcpy(host_batchedHff3[i], device_batchedHff3[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			// cudaMemcpy(host_batchedS[i], device_batchedS[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			// cudaMemcpy(host_batchedS2[i], device_batchedS2[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			// cudaMemcpy(host_batchedV[i], device_batchedV[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			// cudaMemcpy(host_batchedP[i], device_batchedP[i], sizeof(double)* m*n, cudaMemcpyDeviceToHost );
		}
		// // cudaMemcpy(h_A, d_A, sizeof(data_type) * _batch_count * m * n, cudaMemcpyDeviceToHost));
		// cudaMemcpy(h_U, d_U, sizeof(data_type) * _batch_count * m * n, cudaMemcpyDeviceToHost));
		// // cudaMemcpy(h_V, d_V, sizeof(data_type) * _batch_count * m * n, cudaMemcpyDeviceToHost));
		// cudaMemcpy(S, d_S, sizeof(double) * minmn * _batch_count, cudaMemcpyDeviceToHost));
		// cudaMemcpy(info, d_info, sizeof(int) * _batch_count, cudaMemcpyDeviceToHost));

		// // cudaStreamSynchronize(stream));
		// cudaDeviceSynchronize();

		for (int i=0; i < _batch_count; i++) {
			// printf("U[%d]:\n",i);
			// print_matrix(m, n, host_batchedU[i], lda);
			printf("DM[%d]:\n",i);
			print_matrix(m, n, host_batchedDM[i], lda);
			printf("Pot[%d]:\n",i);
			print_matrix(m, n, host_batchedPot[i], lda);
		// 	printf("Hff[%d]:\n",i);
		// 	print_matrix(m, n, host_batchedHff[i], lda);
		// 	printf("Hff2[%d]:\n",i);
		// 	print_matrix(m, n, host_batchedHff2[i], lda);
		// 	printf("Hff3[%d]:\n",i);
		// 	print_matrix(m, n, host_batchedHff3[i], lda);
		// 	// printf("H_A[%d]:\n",i);
		// 	// print_matrix(m, n, h_A +  m * lda * i , 3);
		// 	std::printf("Eigen Values: \n");
		// 	for (int v = 0; v < minmn; v++) {
		// 		std::printf("S0(%d) = %e\n", v + 1, S[i * m + v]);
			// }
		// 	printf("Eigen Vectors:\n");
		// 	print_matrix(m, m, h_V + i * m * lda, ldv);
		// 	printf("S[%d]:\n", i);
		// 	print_matrix(m, m, host_batchedS[i], lda);
		// 	// printf("V[%d]:\n", i);
		// 	// print_matrix(m, m, host_batchedV[i], lda);
		// 	printf("S2[%d]:\n", i);
		// 	print_matrix(m, m, host_batchedS2[i], lda);
		// 	printf("P[%d]:\n", i);
		// 	print_matrix(m, m, host_batchedP[i], lda);
		}
		for (int i = 0; i < _batch_count; i++) {
			// cudaFreeHost(host_batchedU[i]);
			cudaFreeHost(host_batchedDM[i]);
			cudaFreeHost(host_batchedPot[i]);
			// cudaFreeHost(host_batchedHff[i]);
			// cudaFreeHost(host_batchedHff2[i]);
			// cudaFreeHost(host_batchedHff3[i]);
			// cudaFreeHost(host_batchedS[i]);
			// cudaFreeHost(host_batchedS1[i]);
			// cudaFreeHost(host_batchedS2[i]);
			// cudaFreeHost(host_batchedV[i]);
			// cudaFreeHost(host_batchedInvV[i]);
			// cudaFreeHost(host_batchedP[i]);
			cudaFree(device_batchedU[i]); 
			cudaFree(device_batchedDM[i]);
			cudaFree(device_batchedPot[i]);
			// cudaFree(device_batchedHff[i]));
			// cudaFree(device_batchedHff2[i]));
			// cudaFree(device_batchedHff3[i]));
			// cudaFree(device_batchedS[i])); 
			// cudaFree(device_batchedS1[i]));
			// cudaFree(device_batchedS2[i]));
			// cudaFree(device_batchedV[i]));
			// cudaFree(device_batchedInvV[i]));
			// cudaFree(device_batchedP[i]));
		}
		cudaFree(batchedU);
		cudaFree(batchedDM);
		cudaFree(batchedPot);
		// cudaFree(batchedHff));
		// cudaFree(batchedHff2));
		// cudaFree(batchedHff3));
		// cudaFree(batchedS));
		// cudaFree(batchedS1));
		// cudaFree(batchedS2));
		// cudaFree(batchedV));
		// cudaFree(batchedInvV));
		// cudaFree(batchedP));

		
		// cudaFree(d_A));
		// cudaFree(d_U));
		// cudaFree(d_V));
		// cudaFree(d_S));
		// cudaFree(d_info));
		// cudaFree(d_work));

		// cusolverDnDestroyGesvdjInfo(gesvdj_params);
		// cusolverDnDestroy(cusolverH);
		// cudaStreamDestroy(stream));
		cudaDeviceReset();
}

