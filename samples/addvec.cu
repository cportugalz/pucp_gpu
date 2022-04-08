#include<cuda.h>
#include<iostream>

using namespace std;

__global__
void vecAddKernel(float* A,float* B, float* C,int n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<n) C[i] = A[i] + B[i];
}

int main(){
	int device;
	cudaGetDevice(&device);

	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	cout << device;

	

	int n = 100;
	float *A,*B,*C;
	A = new float[n];
	B = new float[n];
	C = new float[n];
	for(int i = 0; i < n; ++i)
		A[i] = B[i] = i;
	float *d_A,*d_B,*d_C;
	int size = n * sizeof(float);
	cudaMalloc((void**)&d_A,size);
	cudaMalloc((void**)&d_B,size);
	cudaMalloc((void**)&d_C,size);
	cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
	vecAddKernel<<<std::ceil(n/20),20>>>(d_A,d_B,d_C,n);
	cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	for(int i = 0; i < n; ++i)
		std::cout << C[i] << " ";
	std::cout << std::endl;
	delete A,B,C;
	return 0;
}
