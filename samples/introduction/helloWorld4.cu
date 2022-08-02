#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iomanip>
#include <iostream>

// __global keyword indicates this methods works on device
__global__ void hello_world_cuda(){	
	printf("%6d%16d%16d%16d%16d%16d\n", blockIdx.x,blockIdx.y,blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
	// 2 block, 10 threads
	printf("**** Hello CUDA **** \n");
	printf("blockIdx.x\tblockIdx.y\tblockIdx.z\tthreadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	dim3 gridDim(2,2,2);
	dim3 blockDim(2,2,2);
	printf("blockIdx.x\tthreadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	hello_world_cuda <<< gridDim, blockDim >>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}