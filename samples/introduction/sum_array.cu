#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
// __global keyword indicates this methods works on device
__global__ void sum_array(int* _a, int* _b, int* _c, int _size){
    int tid = threadIdx.x;
    if (tid < _size){
        _c[tid] = _a[tid] + _b[tid];
    }
}

int main(){
	// 2 block, 10 threads
	printf("**** SUM ARRAY **** \n");
    int h_a[] = {1,2,3,4,5,6,7,8,9,10};
    int h_b[] = {11,12,13,14,15,16,17,18,19,20};
    int* h_c;
    int size = 10;
    int byte_size = size*sizeof(int);
    int* d_a, *d_b, *d_c ;

    cudaMalloc((void**) &d_a, byte_size);
    cudaMalloc((void**) &d_b, byte_size);
    cudaMalloc((void**) &d_c, byte_size);
    clock_t start_time = clock();
    cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);
    clock_t stop_time = clock();
    printf("Time of  host2device: %f s\n", (double)(stop_time - start_time) / CLOCKS_PER_SEC);
    start_time = clock();
    sum_array <<< 1, 10 >>> (d_a, d_b, d_c, size);
    stop_time = clock();
    printf("Time of task: %f s\n", (double)(stop_time - start_time) / CLOCKS_PER_SEC);
    cudaDeviceSynchronize();
    h_c = (int*) malloc(byte_size);
    start_time = clock();
    cudaMemcpy(h_c, d_c, byte_size, cudaMemcpyDeviceToHost);
    stop_time = clock();
    printf("Time of device2host: %f s\n", (double)(stop_time - start_time) / CLOCKS_PER_SEC);
    for (int i=0; i<size; i++){
        printf("%d\t", h_c[i]);
    }
    printf("\n");
	cudaDeviceReset();
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
	return 0;
}