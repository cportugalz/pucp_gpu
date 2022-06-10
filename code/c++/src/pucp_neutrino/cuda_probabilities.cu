#include "cuda_probabilities.h"
#include <cublas_v2.h> 
#include <cuda_runtime.h>
#include "cublas_utils.h"
using namespace std;


// GPU Standard Oscilation
void cuda_StandardOscilation(
	cuDoubleComplex* _U, double* _energy, int _size_data, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double* _events, cuDoubleComplex** _batchedDM){
		cuda_InvisibleDecay(_U, _energy, _size_data, _sigN, _L, _rho, _dm, _alpha, _events, _batchedDM);
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
	// Assigning memory to batched matrices of mass and decay
	cuDoubleComplex** batchedDM = nullptr;
	cuDoubleComplex** batchedPot = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedDM), N*sizeof(cuDoubleComplex *)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedPot), N*sizeof(cuDoubleComplex *)));

	// for (int i=0; i<N; i++) {
	// 	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&batchedDM[i]), 9*sizeof(cuDoubleComplex)));
	// }

	printf("Copying data to GPU\n");
	cudaMemcpy(denergy, henergy, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dalpha, _alpha, 3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ddm, _dm, 2*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dU1, U1, 9*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	// for (int i=0; i<9; i++)
	// 	printf("%f %f\t", U1[i].x, U1[i].y);
	// print_matrix(3,3, U1, 3);
	cuda_StandardOscilation(dU1, denergy, N, _sg, _L, _rho, ddm, dalpha, devents, batchedDM);
	// cublasHandle_t cnpHandle;
    // cublasStatus_t status = cublasCreate(&cnpHandle);
	// cublasDestroy(cnpHandle);
	cudaDeviceSynchronize();
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
	double* _dm, double* _alpha, double* _events, cuDoubleComplex** _batchedDM){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < _size_data){
		cuDoubleComplex Pot[9];
		// double Hff[3][3];
		// double S[3][3];
		// double V[3][3];
		double energy = _energy[tid] * 1e9;
		printf("Calling threadIdx: %d  for energy %.2f\n", tid, energy);
		double rho = _sigN * _rho;
		cuDoubleComplex DM[9] = { {0.0, 0.0} };
		// Matriz de masas y Decay
		// printf("alpha[0]:%f alpha[1]:%f alpha[2]:%f", _alpha[0], _alpha[1], _alpha[2]);
		DM[0] = {0, -0.5 * _alpha[0] / energy};
		DM[4] = {0.5 * _dm[0] / energy, -0.5 * _alpha[1] / energy};
		DM[8] = {0.5 * _dm[1] / energy, -0.5 * _alpha[2] / energy};
		// for (int i=0; i<3; i++){
		// 	for (int j=0;j<3; j++){
		// 		printf("%e %e\t", DM[i+j*3].x, DM[i+j*3].y);
		// 	}
		// 	printf("\n");
		// }
		_batchedDM[tid] = DM;
		printf("ThreadIdx:%d DM[1][1]:%e  %e\n", tid, _batchedDM[tid][4].x, _batchedDM[tid][4].y);
		printf("ThreadIdx:%d DM[2][2]:%e  %e\n", tid, _batchedDM[tid][8].x, _batchedDM[tid][8].y);

		Pot[0] = {rho * 7.63247 * 0.5 * 1.e-14, 0}; Pot[3] = DM[3]; 
		Pot[6]= DM[6]; Pot[1] = DM[1]; Pot[4] = DM[0]; Pot[7] = DM[7];
		Pot[2] = DM[2]; Pot[5] = DM[5]; Pot[8] = DM[0];
		
		// for (int i=0; i<3; i++){
		// 	for (int j=0;j<3; j++){
		// 		printf("%e %e\t", Pot[i+j*3].x, Pot[i+j*3].y);
		// 	}
		// 	printf("\n");
		// }
	}
}


// GPU Invisible Decay
void cuda_InvisibleDecay(
	cuDoubleComplex* _U, double* _energy, int _size_data, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha, double* _events, cuDoubleComplex** _batchedDM) {
		float threads = 1024;
		int blocks = ceil(_size_data/threads);
		printf("Calling Invisible Decay Kernel with %d blocks and %.0f threads per block\n", blocks, threads);
		gpu_invisible_decay<<<blocks, threads>>>(_U, _energy, _size_data, _sigN, _L, _rho, _dm, _alpha, _events, _batchedDM);
		
	// Eigen::MatrixXcd Pot(3, 3);
	// Eigen::MatrixXcd Hff(3, 3);
	// Eigen::MatrixXcd S(3, 3);
	// Eigen::MatrixXcd V(3, 3);
	// double energy = _energy * 1e9;
	// double rho = _sigN * _rho;
    // std::complex<double> DM[3][3] = { std::complex<double>(0, 0) };
	// /* Matriz de masas y Decay */
	// DM[0][0] = std::complex<double>(0, -0.5 * _alpha[0] / energy);
	// DM[1][1] = std::complex<double>(0.5 * _dm[0] / energy, -0.5 * _alpha[1] / energy);
	// DM[2][2] = std::complex<double>(0.5 * _dm[1] / energy, -0.5 * _alpha[2] / energy);
	// Pot << std::complex<double>(rho * 7.63247 * 0.5 * 1.e-14, 0), DM[0][1], DM[0][2],
	// 	DM[1][0], DM[0][0], DM[1][2],
	// 	DM[2][0], DM[2][1], DM[0][0];
	// /* Inicializando las matrices para Eigen */
	// Eigen::MatrixXcd UPMNS(3, 3);
	// UPMNS << _U[0][0], _U[0][1], _U[0][2],
	// 	_U[1][0], _U[1][1], _U[1][2],
	// 	_U[2][0], _U[2][1], _U[2][2];
	// Eigen::MatrixXcd Hd(3, 3);
	// Hd << DM[0][0], DM[0][1], DM[0][2],
	// 	DM[1][0], DM[1][1], DM[1][2],
	// 	DM[2][0], DM[2][1], DM[2][2];
	// /* Hamiltoniano final efectivo */
	// Hff = UPMNS * Hd * UPMNS.adjoint() + Pot;
	// /* Calculando los autovalores y autovectores */
	// Eigen::ComplexEigenSolver<Eigen::MatrixXcd> tmp;
	// tmp.compute(Hff);
	// /* Calculamos la matriz S y ordenamos los autovalores */
	// V = tmp.eigenvectors();
	// S << exp(-ProbConst::I * tmp.eigenvalues()[0] * _L * 1.e9 / ProbConst::GevkmToevsq), DM[0][0], DM[0][0],
	// 	DM[0][0], exp(-ProbConst::I * tmp.eigenvalues()[1] * _L * 1.e9 / ProbConst::GevkmToevsq), DM[0][0],
	// 	DM[0][0], DM[0][0], exp(-ProbConst::I * tmp.eigenvalues()[2] * _L * 1.e9 / ProbConst::GevkmToevsq);
	// S = (V)*S * (V.inverse());
	// // Calculando la matriz de probabilidad
	// for (int i = 0; i < 3; i++)
	// {
	// 	for (int j = 0; j < 3; j++)
	// 	{
	// 		_P[i][j] = abs(S.col(i)[j] * S.col(i)[j]);
	// 	}
	// } 
}


const std::complex<double> ProbConst::I = std::complex<double>(0, 1);
const std::complex<double> ProbConst::Z0 = std::complex<double>(0, 0);
const double ProbConst::hbar = 6.58211928*1.e-25;
const double ProbConst::clight = 299792458;
//double GevkmToevsq = hbar*clight*1.e15;
const double ProbConst::GevkmToevsq = 0.197327; // Approximated value


// Working on host code the matrix U but returning it as column form cuDoubleComplex
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
	for(int i=0; i<3; i++){
		for(int j=0; j<3;j++){
			// printf("%f %f\t", U[i][j].real(), U[i][j].imag());
			cuU[i+j*3] = make_cuDoubleComplex(U[i][j].real(), U[i][j].imag());
		}
		// printf("\n");
	}
	return cuU;
}
