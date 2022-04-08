#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>


__global__ void access_data(int* _device_data){
	printf("%6d%16d%16d%16d%16d%16d%10d\n", blockIdx.x,blockIdx.y,blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, _device_data[threadIdx.x]);
}

__global__ void access_data_mapped(int* _device_data ){
	printf("%6d%16d%16d%16d%16d%16d%10d\n", blockIdx.x,blockIdx.y,blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, _device_data[blockIdx.x*blockDim.x + threadIdx.x]);
}
// Grids -> Blocks -> Threads
int main(){
	int data_size = 8;
	int data_bytes_size = sizeof(int)*data_size;
	int host_data [] = {23, 34, 45, 66, 45, 67, 77, 1};
	printf("Host data:");
	for (int i=0; i<data_size;i++){
		printf("%d ", host_data[i]);
	}
	printf("\n");
	int* device_data;
	cudaMalloc((void**)&device_data, data_bytes_size);
	cudaMemcpy(device_data, host_data, data_bytes_size, cudaMemcpyHostToDevice);
	printf("1 block per 8 threads - One dimension\n");
	printf("blockIdx.x\tblockIdx.y\tblockIdx.z\tthreadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	access_data <<<1, data_size>>>(device_data);
	cudaDeviceSynchronize();
	printf("2 block per 8 threads - Unnecessary\n");
	printf("blockIdx.x\tblockIdx.y\tblockIdx.z\tthreadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	access_data <<<2, data_size>>>(device_data);
	cudaDeviceSynchronize();
	printf("2 block per 4 threads - Bad index\n");
	printf("blockIdx.x\tblockIdx.y\tblockIdx.z\tthreadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	access_data <<<2, 4>>>(device_data);
	cudaDeviceSynchronize();
	printf("2 block per 4 threads - Mapping indexes \n");
	printf("blockIdx.x\tblockIdx.y\tblockIdx.z\tthreadIdx.x\tthreadIdx.y\tthreadIdx.z\n");
	access_data_mapped <<<2, 4>>>(device_data);
	cudaDeviceReset();
	return 0;
}