#include "cuda_probabilities.h"
#include <cuda_runtime.h>


void cuda_simulation_StandardOscilation(
	int _num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho,
	double* _dm, double* _alpha){
	int N = _num_simulations;
	// for standard oscilation, invisible decay and nsi
	clock_t start_time = clock();
	cuDoubleComplex* U1 = make_umns(_sg, _th, _dcp);
	double* henergy = (double*) malloc(N * sizeof(double));
	printf("Creating %d values of energy on host.\n", N);
	for(int iter_energy=0; iter_energy<N; ) {
		henergy[iter_energy] = (iter_energy+1)/100.0;
		iter_energy += 1;
	}
	double* denergy = nullptr;
	double *devents = nullptr;
	double* dalpha = nullptr;
	cuDoubleComplex* dU1 = nullptr;
	double* ddm = nullptr;
	printf("Assigning memory on device.\n");
	cudaMalloc(reinterpret_cast<void **>(&denergy), N*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&dalpha), 3*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&ddm), 2*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&dU1), 9*sizeof(cuDoubleComplex));
	printf("Copying values of energy from host to device.\n");
	cudaMemcpy(denergy, henergy, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dalpha, _alpha, 3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ddm, _dm, 2*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dU1, U1, 9*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cuda_StandardOscilation(dU1, denergy, N, _sg, _L, _rho, ddm, dalpha, devents);
	cudaFree(denergy);
	cudaFree(devents);
	cudaFree(dalpha);
	cudaFree(ddm);
	cudaFree(dU1);
	free(henergy);
	delete[] U1;
	clock_t stop_time = clock();
	printf("Total time: %.7fs\n", (double)(stop_time - start_time)/CLOCKS_PER_SEC);
}

void cuda_simulation_InvisibleDecay(
	int _num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho,
	double* _dm, double* _alpha){
	int N = _num_simulations;
	// for standard oscilation, invisible decay and nsi
	clock_t start_time = clock();
	cuDoubleComplex* U1 = make_umns(_sg, _th, _dcp);
	double* henergy = (double*) malloc(N * sizeof(double));
	printf("Creating %d values of energy on host.\n", N);
	for(int iter_energy=0; iter_energy<N; ) {
		henergy[iter_energy] = (iter_energy+1)/100.0;
		iter_energy += 1;
	}
	double* denergy = nullptr;
	double *devents = nullptr;
	double* dalpha = nullptr;
	cuDoubleComplex* dU1 = nullptr;
	double* ddm = nullptr;
	printf("Assigning memory on device.\n");
	cudaMalloc(reinterpret_cast<void **>(&denergy), N*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&dalpha), 3*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&ddm), 2*sizeof(double));
	cudaMalloc(reinterpret_cast<void **>(&dU1), 9*sizeof(cuDoubleComplex));
	printf("Copying values of energy from host to device.\n");
	cudaMemcpy(denergy, henergy, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dalpha, _alpha, 3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ddm, _dm, 2*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dU1, U1, 9*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cuda_InvisibleDecay(dU1, denergy, N, _sg, _L, _rho, ddm, dalpha, devents);
	cudaFree(denergy);
	cudaFree(devents);
	cudaFree(dalpha);
	cudaFree(ddm);
	cudaFree(dU1);
	free(henergy);
	delete[] U1;
	clock_t stop_time = clock();
	printf("Total time: %.7fs\n", (double)(stop_time - start_time)/CLOCKS_PER_SEC);
}