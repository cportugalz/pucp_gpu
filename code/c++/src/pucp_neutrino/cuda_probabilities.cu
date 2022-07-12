#include "cuda_probabilities.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"
#include <cusolverDn.h>
using namespace std;


// GPU Standard Oscilation
void cuda_StandardOscilation(
	cuDoubleComplex* _U, double* _energy, int _size_data, int _sigN, double _L, double _rho,
	double* _dm, double* _alpha, double* _events){
		cuda_InvisibleDecay(_U, _energy, _size_data, _sigN, _L, _rho, _dm, _alpha, _events);
}


void cuda_simulation_StandardOscilation(
	int _num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho,
	double* _dm, double* _alpha){
	int N = _num_simulations;
	// for standard oscilation, invisible decay and nsi
	cuDoubleComplex* U1 = make_umns(_sg, _th, _dcp);
	double* henergy = (double*)malloc(N*sizeof(double));
	printf("Creating %d values of energy for simulation\n", N);
	for(int iter_energy=0; iter_energy<=N; ) {
		double energy = (iter_energy+1)/100.0;
		henergy[iter_energy] = energy;
		iter_energy += 1;
	}
	printf("Assigning device memory\n");
	double* denergy = nullptr;
	double *devents = nullptr;
	double* dalpha = nullptr;
	cuDoubleComplex* dU1 = nullptr;
	double* ddm = nullptr;
	cudaMalloc(reinterpret_cast<void **>(&denergy), N*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&dalpha), 3*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&ddm), 2*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&dU1), 9*sizeof(cuDoubleComplex));

	printf("Copying data to GPU\n");
	cudaMemcpy(denergy, henergy, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dalpha, _alpha, 3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ddm, _dm, 2*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dU1, U1, 9*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cuda_StandardOscilation(dU1, denergy, N, _sg, _L, _rho, ddm, dalpha, devents);
	cudaDeviceReset();
	// cudaFree(denergy);
	// cudaFree(devents);
	// cudaFree(dalpha);
	// cudaFree(ddm);
	// cudaFree(dU1);
	free(henergy);
	free(U1);
}

// GPU Kernel for visible decay
__global__ void gpu_invisible_decay(
	cuDoubleComplex* _U, double* _energy, int _size_data, int _sigN, double _L, double _rho,
	double* _dm, double* _alpha, double* _events, cuDoubleComplex** _batchedDM,
	cuDoubleComplex** _batchedPot, cuDoubleComplex** _batchedU){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < _size_data){
		//Copying _size_data of U to batchedU 
		for (int i=0; i<9; i++) {
			_batchedU[tid][i] = _U[i];
		}
		double energy = _energy[tid] * 1e9;
		// printf("Calling threadIdx: %d  for energy %e\n", tid, energy);
		double rho = _sigN * _rho;
		// Matriz de masas y Decay
		_batchedDM[tid][0] = {0, -0.5 * _alpha[0] / energy};
		_batchedDM[tid][4] = {0.5 * _dm[0] / energy, -0.5 * _alpha[1] / energy};
		_batchedDM[tid][8] = {0.5 * _dm[1] / energy, -0.5 * _alpha[2] / energy};
		
		_batchedPot[tid][0] = {rho * 7.63247 * 0.5 * 1.e-14, 0}; _batchedPot[tid][3] = _batchedDM[tid][3];
		_batchedPot[tid][6]= _batchedDM[tid][6]; _batchedPot[tid][1] = _batchedDM[tid][1]; _batchedPot[tid][4] = _batchedDM[tid][0];
		_batchedPot[tid][7] = _batchedDM[tid][7]; _batchedPot[tid][2] = _batchedDM[tid][2]; _batchedPot[tid][5] = _batchedDM[tid][5]; 
		_batchedPot[tid][8] = _batchedDM[tid][0];
	}
}


__global__ void sum_batched(
	cuDoubleComplex** Pot, cuDoubleComplex** Hff2, cuDoubleComplex** Hff3, cuDoubleComplex* d_A, int _batch_count) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	//   0		1
	// 1 4 7	4 7 10
	// 2 5 8    5 8 11
	// 3 6 9    6 9 12
	// Pot =  { { 1,2,3,4,5,6,7,8,9}, {4,5,6,7,8,9,10,11,12}, {}, {} }
	// Hff2 = { { 1,2,3,4,5,6,7,8,9}, {4,5,6,7,8,9,10,11,12}, {}, {} }
	// Hff3 = { { 2,4,6,8, 10 ...  }, {...}, { ...}, { ... } }
	if (tid < _batch_count) {
		for (int i=0; i<9; i++) {
				Hff3[tid][i].x = Hff2[tid][i].x + Pot[tid][i].x;
				Hff3[tid][i].y = Hff2[tid][i].y + Pot[tid][i].y;
				// d_A[tid * 9 + i].x =  Hff2[tid][i].x + Pot[tid][i].x;
				// d_A[tid * 9 + i].y =  Hff2[tid][i].y + Pot[tid][i].y;
				(d_A + tid * 9 + i)->x =  10;
				(d_A + tid * 9 + i)->y =  10;
				printf("[%d ]%e %ej \n", tid * 9 + i, d_A[tid * 9 + i].x, d_A[tid * 9 + i].y);
		}
	}
}

// GPU Invisible Decay
void cuda_InvisibleDecay(
	cuDoubleComplex* _U, double* _energy, int _batch_count, int _sigN, double _L, double _rho,
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
		// Assigning memory to batched matrices of mass and decay
		data_type** batchedU = nullptr;
		data_type* host_batchedU[_batch_count];
		data_type* device_batchedU[_batch_count];

		data_type** batchedDM = nullptr;
		data_type* host_batchedDM[_batch_count];
		data_type* device_batchedDM[_batch_count];

		data_type** batchedPot = nullptr;
		data_type* host_batchedPot[_batch_count];
		data_type* device_batchedPot[_batch_count];

		data_type** batchedHff = nullptr;
		data_type* host_batchedHff[_batch_count];
		data_type* device_batchedHff[_batch_count];

		data_type** batchedHff2 = nullptr;
		data_type* host_batchedHff2[_batch_count];
		data_type* device_batchedHff2[_batch_count];

		data_type** batchedHff3 = nullptr;
		data_type* host_batchedHff3[_batch_count];
		data_type* device_batchedHff3[_batch_count];

		for (int i = 0; i < _batch_count; i++) {
			host_batchedU[i]   = (data_type*) malloc(sizeof(data_type)*m*n);
			host_batchedDM[i]  = (data_type*) malloc(sizeof(data_type)*m*n);
			host_batchedPot[i] = (data_type*) malloc(sizeof(data_type)*m*n);
			host_batchedHff[i] = (data_type*) malloc(sizeof(data_type)*m*n);
			host_batchedHff2[i] = (data_type*) malloc(sizeof(data_type)*m*n);
			host_batchedHff3[i] = (data_type*) malloc(sizeof(data_type)*m*n);

			for (int j = 0; j < m*n; j++){
				host_batchedU[i][j]  = make_cuDoubleComplex(0.0,0.0);
				host_batchedDM[i][j]  = make_cuDoubleComplex(0.0,0.0);
				host_batchedPot[i][j] = make_cuDoubleComplex(0.0,0.0);
				host_batchedHff[i][j] = make_cuDoubleComplex(0.0,0.0);
				host_batchedHff2[i][j] = make_cuDoubleComplex(0.0,0.0);
				host_batchedHff3[i][j] = make_cuDoubleComplex(0.0,0.0);
			}
		}
		for (int i = 0; i < _batch_count; i++) {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_batchedU[i]), sizeof(data_type) * m*n));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_batchedDM[i]), sizeof(data_type) * m*n));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_batchedPot[i]), sizeof(data_type) * m*n));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_batchedHff[i]), sizeof(data_type) * m*n));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_batchedHff2[i]), sizeof(data_type) * m*n));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_batchedHff3[i]), sizeof(data_type) * m*n));
		}
	
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedU),  	_batch_count*sizeof(data_type *)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedDM),  	_batch_count*sizeof(data_type *)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedPot), 	_batch_count*sizeof(data_type *)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedHff), 	_batch_count*sizeof(data_type *)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedHff2), 	_batch_count*sizeof(data_type *)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedHff3), 	_batch_count*sizeof(data_type *)));
		for (int i = 0; i < _batch_count; i++) {
			CUDA_CHECK(cudaMemcpy(device_batchedU[i], host_batchedU[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(device_batchedDM[i], host_batchedDM[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(device_batchedPot[i], host_batchedPot[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(device_batchedHff[i], host_batchedHff[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(device_batchedHff2[i], host_batchedHff2[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(device_batchedHff3[i], host_batchedHff3[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
		}
		cudaMemcpy(batchedU, device_batchedU, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		cudaMemcpy(batchedDM, device_batchedDM, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		cudaMemcpy(batchedPot, device_batchedPot, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		cudaMemcpy(batchedHff, device_batchedHff, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		cudaMemcpy(batchedHff2, device_batchedHff2, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);
		cudaMemcpy(batchedHff3, device_batchedHff3, sizeof(data_type*)*_batch_count, cudaMemcpyHostToDevice);

		printf("Calling Invisible Decay Kernel with %d blocks and %.0f threads per block\n", blocks, threads);
		gpu_invisible_decay<<<blocks, threads>>>(
			_U, _energy, _batch_count, _sigN, _L, _rho, _dm, _alpha, _events,
			batchedDM, batchedPot, batchedU);
		cudaDeviceSynchronize();
		// CuBlas Operations
		cublasHandle_t cublasH = NULL;
		// cudaStream_t stream = NULL;
		
		const data_type alpha = {1.0, 0.0};
		const data_type beta = {0.0, 0.0};
		
		CUBLAS_CHECK(cublasCreate(&cublasH));
		// CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		// CUBLAS_CHECK(cublasSetStream(cublasH, stream));
		CUBLAS_CHECK(cublasZgemmBatched(
			cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (const data_type * const *)batchedU, lda,
			(const data_type*  const*) batchedDM, ldb, &beta, batchedHff, ldc, _batch_count));
		// CUDA_CHECK(cudaStreamSynchronize(stream));
		CUBLAS_CHECK(cublasZgemmBatched(
			cublasH, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, &alpha, (const data_type * const *)batchedHff, lda,
			(const data_type*  const*) batchedU, ldb, &beta, batchedHff2, ldc, _batch_count));
		// CUDA_CHECK(cudaStreamSynchronize(stream));
		// CUBLAS_CHECK(cublasDestroy(cublasH));

		// Calculating eigen value with cuSolver
		cusolverDnHandle_t cusolverH = NULL;
		gesvdjInfo_t gesvdj_params = NULL;
		// cuDoubleComplex *h_A = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * _batch_count * m * lda);
		data_type *d_A = nullptr;    /* lda-by-m-by-batchSize */
		data_type *h_A = (data_type*) malloc(sizeof(data_type) * _batch_count * m * n);    /* lda-by-m-by-batchSize */
		// data_type *d_U = nullptr;    /* lda-by-m-by-batchSize */
		// data_type *h_U = (data_type*) malloc( sizeof(data_type) * ldu * m * _batch_count);    /* lda-by-m-by-_batch_count */
		// data_type *d_V = nullptr;    /* lda-by-m-by-_batch_count */
		// data_type *h_V = (data_type*) malloc( sizeof(data_type) * ldv * n * _batch_count);    /* lda-by-m-by-_batch_count */
		// double* S = (double*) malloc(sizeof(double) * minmn * _batch_count);
		// double *d_S = nullptr; /* minmn-by-batchSize */
		// int* info = (int*) malloc(sizeof(int) * _batch_count);
		// int *d_info = nullptr; /* batchSize */


		// int lwork = 0;            /* size of workspace */
		// data_type *d_work = nullptr; /* device workspace for getrf */

		// const double tol = 1.e-7;
		// const int max_sweeps = 100;
		// const int sort_svd = 0;                                  /* don't sort singular values */
		// const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * _batch_count * m * n));
		sum_batched<<<blocks, threads>>> (batchedPot, batchedHff2, batchedHff3, d_A, _batch_count);
		cudaDeviceSynchronize();
		CUDA_CHECK(
			cudaMemcpy(h_A, d_A, sizeof(data_type) * _batch_count * m * n, cudaMemcpyDeviceToHost));
		// Hff[0] = U[0] * DM[0] * UC[0] + Pot[0]
		// Hff[1] = U[1] * DM[1] * UC[1] + Pot[1]
		// Hff[N-1] = U[N-1] * DM[N-1] * UC[N-1] + Pot[N-1]
		/* step 1: create cusolver handle, bind a stream */
		// cusolverDnCreate(&cusolverH);
		// cusolverDnSetStream(cusolverH, stream);

		/* step 2: configuration of syevj */
		// cusolverDnCreateGesvdjInfo(&gesvdj_params);

		/* default value of tolerance is machine zero */
		// cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);

		/* default value of max. sweeps is 100 */
		// cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
		// Pot Sum and copying to d_A of 1D with _batch_count * m * lda
		

		/* disable sorting */
		// cusolverDnXgesvdjSetSortEig(gesvdj_params, sort_svd));
		// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * ldu * m * _batch_count));
		// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(data_type) * ldv * n * _batch_count));
		// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) *minmn * _batch_count));
		// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) *_batch_count));

		
		// cusolverDnZgesvdjBatched_bufferSize(cusolverH, jobz, m, n, d_A, lda, d_S, d_U,
		// 	ldu, d_V, ldv, &lwork, gesvdj_params,
		// 	_batch_count);
		// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));
		
		
		// cusolverDnZgesvdjBatched(cusolverH, jobz, m, n, d_A, lda, d_S, d_U, ldu, d_V,
		// 	ldv, d_work, lwork, d_info, gesvdj_params, _batch_count);

		for (int i = 0; i < _batch_count; i++) {
			cudaMemcpy(host_batchedU[i], device_batchedU[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedDM[i], device_batchedDM[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedPot[i], device_batchedPot[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedHff[i], device_batchedHff[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedHff2[i], device_batchedHff2[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedHff3[i], device_batchedHff3[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
		}
		// CUDA_CHECK(
		// 	cudaMemcpy(h_A, d_A, sizeof(data_type) * _batch_count * m * n, cudaMemcpyDeviceToHost));
		// CUDA_CHECK(
		// 	cudaMemcpy(h_U, d_U, sizeof(data_type) * _batch_count * m * n, cudaMemcpyDeviceToHost));
		// CUDA_CHECK(
		// 	cudaMemcpy(h_V, d_V, sizeof(double) * _batch_count * m, cudaMemcpyDeviceToHost));
		// CUDA_CHECK(
		// 	cudaMemcpy(S, d_S, sizeof(int) * _batch_count, cudaMemcpyDeviceToHost));
		// CUDA_CHECK(
		// 	cudaMemcpy(info, d_info, sizeof(int) * _batch_count, cudaMemcpyDeviceToHost));

		// CUDA_CHECK(cudaStreamSynchronize(stream));	
		cudaDeviceSynchronize();

		for (int i=0; i < _batch_count; i++) {
			printf("U[%d]:\n",i);
			print_matrix(m, n, host_batchedU[i], lda);
			printf("DM[%d]:\n",i);
			print_matrix(m, n, host_batchedDM[i], lda);
			printf("Pot[%d]:\n",i);
			print_matrix(m, n, host_batchedPot[i], lda);
			printf("Hff[%d]:\n",i);
			print_matrix(m, n, host_batchedHff[i], lda);
			printf("Hff2[%d]:\n",i);
			print_matrix(m, n, host_batchedHff2[i], lda);
			printf("Hff3[%d]:\n",i);
			print_matrix(m, n, host_batchedHff3[i], lda);
			printf("H_A[%d]:\n",i);
			print_matrix(m, n, h_A + _batch_count * m * lda + i , 3);
			// if (0 == info[i]) {
			// 	std::printf("matrix %d: gesvdj converges \n", i);
			// } else if (0 > info[i]) {
			// 	/* only info[0] shows if some input parameter is wrong.
			// 	* If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
			// 	*/
			// 	std::printf("Error: %d-th parameter is wrong \n", -info[i]);
			// 	exit(1);
			// } else { /* info = m+1 */
			// 		/* if info[i] is not zero, Jacobi method does not converge at i-th matrix. */
			// 	std::printf("WARNING: matrix %d, info = %d : gesvdj does not converge \n", i, info[i]);
			// }
			// std::printf("Eigen Values: \n");
			// for (int i = 0; i < minmn; i++) {
			// 	std::printf("S0(%d) = %e\n", i + 1, S[_batch_count * m + i]);
			// }
			
			// printf("Eigen Vectors:\n");
			// print_matrix(m,m, &h_V[_batch_count * m * lda], lda);
			// printf("==== \n");
		}
		CUDA_CHECK(cudaFree(d_A));
		// CUDA_CHECK(cudaFree(d_U));
		// CUDA_CHECK(cudaFree(d_V));
		// CUDA_CHECK(cudaFree(d_S));
		// CUDA_CHECK(cudaFree(d_info));
		// CUDA_CHECK(cudaFree(d_work));
		
		// cusolverDnDestroyGesvdjInfo(gesvdj_params);
		// cusolverDnDestroy(cusolverH);
		// CUDA_CHECK(cudaStreamDestroy(stream));

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
	return cuU;
}
