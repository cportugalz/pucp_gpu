#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iomanip>
#include <iostream>

// __global keyword indicates this methods works on device
__global__ void hello_world_cuda(){	
	printf("%6d\t %12d\t %12d \n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
	// 1 block, 10 threads
	printf("**** Hello CUDA **** \n");
	printf("threadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	hello_world_cuda <<< 1, 10 >>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}