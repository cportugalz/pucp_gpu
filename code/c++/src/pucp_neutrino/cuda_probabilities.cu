#include "cuda_probabilities.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"
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
	cuDoubleComplex** Pot, cuDoubleComplex** Hff2, cuDoubleComplex** Hff3, int _batch_count) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < _batch_count) {
		for (int i=0; i<9; i++) {
				Hff3[tid][i].x = Hff2[tid][i].x + Pot[tid][i].x;
				Hff3[tid][i].y = Hff2[tid][i].y + Pot[tid][i].y;
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
		cudaStream_t stream = NULL;
		
		const data_type alpha = {1.0, 1.0};
		const data_type beta = {0.0, 0.0};
		
		CUBLAS_CHECK(cublasCreate(&cublasH));
		CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		CUBLAS_CHECK(cublasSetStream(cublasH, stream));
		CUBLAS_CHECK(cublasZgemmBatched(
			cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (const data_type * const *)batchedU, lda,
			(const data_type*  const*) batchedDM, ldb, &beta, batchedHff, ldc, _batch_count));
		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUBLAS_CHECK(cublasZgemmBatched(
			cublasH, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, &alpha, (const data_type * const *)batchedHff, lda,
			(const data_type*  const*) batchedU, ldb, &beta, batchedHff2, ldc, _batch_count));
		CUDA_CHECK(cudaStreamSynchronize(stream));
		// Pot Sum
		sum_batched<<<blocks, threads>>> (batchedPot, batchedHff2, batchedHff3, _batch_count);
		cudaDeviceSynchronize();

		for (int i = 0; i < _batch_count; i++) {
			cudaMemcpy(host_batchedU[i], device_batchedU[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedDM[i], device_batchedDM[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedPot[i], device_batchedPot[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedHff[i], device_batchedHff[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedHff2[i], device_batchedHff2[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
			cudaMemcpy(host_batchedHff3[i], device_batchedHff3[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
		}
		CUDA_CHECK(cudaStreamSynchronize(stream));
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
		}
		CUBLAS_CHECK(cublasDestroy(cublasH));
		CUDA_CHECK(cudaStreamDestroy(stream));
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
