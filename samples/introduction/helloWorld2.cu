#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iomanip>
#include <iostream>

// __global keyword indicates this methods works on device
__global__ void hello_world_cuda(){	
	printf("%6d%16d%16d%16d\n", blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
	// 2 block, 10 threads
	printf("**** Hello CUDA **** \n");
	printf("blockIdx.x\tthreadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	hello_world_cuda <<< 2, 10 >>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}