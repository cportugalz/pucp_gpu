#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

__global__ void sum_vector(int* a, int*b, double* c, int size, int* sum){

	int tid = threadIdx.x;
	if (tid < size){
		c[tid] = double(a[tid]) + double(b[tid]);
	}
}


int main(){
	printf("**** Test Program ****\n");
	int data1[] = {1,2,3,4,5,6,7,8,9,10};
	int data2[] = {4,5,6,7,8,9,9,20,1,2};

	int size = 10;
	double* data3 = (double*) malloc(size * sizeof(double));

	int sum = 0;
	for (int i=0; i<size; i++){
		sum += data1[i] + data2[i];
	}
	printf("CPU sum:%d\n", sum);
	int* ddata1;
	int* ddata2;
	double* ddata3;
	double dsum = 0;
	// assigning memory
	cudaMalloc((void**)&ddata1, size*sizeof(int));
	cudaMalloc((void**)&ddata2, size*sizeof(int));
	cudaMalloc((void**)&ddata3, size*sizeof(double));
	
	//copying data
	cudaMemcpy(ddata1, data1, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ddata2, data2, size*sizeof(int), cudaMemcpyHostToDevice);
	sum_vector<<< 1, size >>>(ddata1, ddata2, ddata3, size, &sum);
	cudaMemcpy(data3, ddata3, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i< size; i++){
		printf("%f\t", data3[i]);
	}
	cublasHandle_t cublasH = NULL;
	cublasCreate(&cublasH);
	cublasDasum(cublasH, size, ddata3, 1, &dsum);
	cudaDeviceSynchronize();
	cublasDestroy(cublasH);
	printf("\nGPU Sum:%f\n", dsum);
	cudaDeviceReset();	
	cudaFree(ddata1);
	cudaFree(ddata2);
	cudaFree(ddata3);
	free(data3);

}

