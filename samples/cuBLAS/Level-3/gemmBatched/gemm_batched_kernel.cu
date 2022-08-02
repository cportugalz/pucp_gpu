
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

__global__ void init_data(data_type** A, data_type** B, int _size){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	for (int i=0; i<_size; i++){
		A[tid][i] = 5;
		B[tid][i] = 6;
	}
}


int main(int argc, char *argv[]) {
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;


	const int m = 3;
	const int n = 3;
	const int k = 3;
	const int lda = 3;
	const int ldb = 3;
	const int ldc = 3;
	const int batch_count = 3;

	const data_type alpha = 1.0;
	const data_type beta = 0.0;


	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_N;


	/* step 1: create cublas handle, bind a stream */
	CUBLAS_CHECK(cublasCreate(&cublasH));
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUBLAS_CHECK(cublasSetStream(cublasH, stream));
	data_type* h_A[batch_count];
	data_type* h_B[batch_count];
	data_type* h_C[batch_count];
	for (int i=0; i<batch_count; i++) {
		h_A[i] = (data_type*) malloc(sizeof(data_type)*m*n);
		h_B[i] = (data_type*) malloc(sizeof(data_type)*m*n);
		h_C[i] = (data_type*) malloc(sizeof(data_type)*m*n);

		for (int j=0;j<4; j++){
			h_A[i][j] = 0;
			h_B[i][j] = 0;
			h_C[i][j] = 0;
		}
	}
	data_type* d_A[batch_count];
	data_type* d_B[batch_count];
	data_type* d_C[batch_count];
	for (int i = 0; i < batch_count; i++) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A[i]), sizeof(data_type) * m*n));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B[i]), sizeof(data_type) * m*n));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(data_type) * m*n));
	}
	data_type** d_A_array;
	data_type** d_B_array;
	data_type** d_C_array;
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(data_type *) * batch_count));
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(data_type *) * batch_count));
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(data_type *) * batch_count));

	for (int i = 0; i < batch_count; i++) {
		CUDA_CHECK(cudaMemcpy(d_A[i], h_A[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_B[i], h_B[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_C[i], h_C[i], sizeof(data_type) * m*n, cudaMemcpyHostToDevice));
	}
	cudaMemcpy(d_A_array, d_A, sizeof(data_type*)*batch_count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_array, d_B, sizeof(data_type*)*batch_count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_array, d_C, sizeof(data_type*)*batch_count, cudaMemcpyHostToDevice);

	init_data<<<1, batch_count>>>(d_A_array, d_B_array, n*m);
	cudaDeviceSynchronize();
	
	
	for (int i = 0; i < batch_count; i++) {
		cudaMemcpy(h_A[i], d_A[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
		cudaMemcpy(h_B[i], d_B[i], sizeof(data_type)* m*n, cudaMemcpyDeviceToHost );
	}
	
	print_matrix(m, k, h_A[0], lda);
	print_matrix(m, k, h_A[1], lda);
	print_matrix(m, k, h_B[0], lda);
	print_matrix(m, k, h_B[1], lda);

	// CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * batch_count,
	//                            cudaMemcpyHostToDevice,stream));

	CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
	                                d_B_array, ldb, &beta, d_C_array, ldc, batch_count));

	/* step 4: copy data to host */
	for (int i = 0; i < batch_count; i++) {
	    CUDA_CHECK(cudaMemcpyAsync(h_C[i], d_C[i], sizeof(data_type) * m*n,
	                               cudaMemcpyDeviceToHost, stream));
	}
	CUDA_CHECK(cudaStreamSynchronize(stream));



	/*
	 *   C = | 19.0 | 22.0 | 111.0 | 122.0 |
	 *       | 43.0 | 50.0 | 151.0 | 166.0 |
	 */

	printf("C[0]\n");
	print_matrix(m, n, h_C[0], ldc);
	printf("=====\n");

	printf("C[1]\n");
	print_matrix(m, n, h_C[1], ldc);
	printf("=====\n");

	/* free resources */
	// CUDA_CHECK(cudaFree(d_A_array));
	// CUDA_CHECK(cudaFree(d_B_array));
	// CUDA_CHECK(cudaFree(d_C_array));
	// for (int i = 0; i < batch_count; i++) {
	//     CUDA_CHECK(cudaFree(d_A_array[i]));
	//     CUDA_CHECK(cudaFree(d_B_array[i]));
	//     CUDA_CHECK(cudaFree(d_C_array[i]));
	// }

	CUBLAS_CHECK(cublasDestroy(cublasH));


	CUDA_CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}
