#include<cuda.h>
#include<vector>
#include<iostream>
#include<ctime>
#define TILE_WIDTH 16

using namespace std;

typedef int num;
typedef vector<num> vec_num;
typedef vector<vec_num> mat_num;

__global__ void simpleMatrixMulKernel(num* M, num* N, num* P,int Width) {
	// Calculate the row index of the P element and M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of P and N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < Width) && (Col < Width)) {
		num Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < Width; ++k) {
			Pvalue += M[Row*Width+k]*N[k*Width+Col];
		}
		P[Row*Width+Col] = Pvalue;
	}
}

__global__ void MatrixMulKernel(num* d_M, num* d_N, num* d_P, int Width) {

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	num Pvalue = 0;
	// Loop over the d_M and d_N tiles required to compute d_P element
	for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {

	// Collaborative loading of d_M and d_N tiles into shared memory
		if ((Row< Width) && (ph*TILE_WIDTH+tx)< Width)
			Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
		if ((ph*TILE_WIDTH+ty)<Width && Col<Width)
			Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	if ((Row<Width) && (Col<Width)) d_P[Row*Width + Col] = Pvalue;
}

void fill(num* M, int size){
	for(int i = 0; i < size; i ++)
		M[i] = rand()%100 + 1;
}

void printMatrix(num* M, int size){
	for(int i = 0; i < size; ++i){
		for(int j = 0; j < size; ++j)
			cout << M[i*size + j] << " ";
		cout << '\n';
	}
}

int main(int argc, char** argv){
	int m,n,k;
	if(argc > 1){
		m = atoi(argv[1]);
		n = atoi(argv[2]);
		k = atoi(argv[3]);
	}
	else{
		m = n = k = 32;
	}
	
	num M1[m*n],M2[n*k],MR[n*n];
	fill(M1,m*n);
	fill(M2,n*k);

	num* d_M1 ;//= &M1[0];
	num* d_M2 ;//= &M2[0];
	num* d_MR ;//= &MR[0];
	cout << "MAtrix M1:" << '\n';
	printMatrix(M1,n);
	cout << "MAtrix M2:" << '\n';
	printMatrix(M2,n);

	cudaMalloc((void**)&d_M1,m*n*sizeof(num));
	cudaMemcpy(d_M1,M1,m*n*sizeof(num),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_M2,n*k*sizeof(num));
	cudaMemcpy(d_M2,M2,n*k*sizeof(num),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_MR,n*n*sizeof(num));
	cudaMemcpy(d_MR,MR,n*n*sizeof(num),cudaMemcpyHostToDevice);

	const dim3 blockSize(TILE_WIDTH,TILE_WIDTH,1);
	const dim3 gridSize(ceil(n/TILE_WIDTH),ceil(n/TILE_WIDTH),1);
	
	clock_t start;
	double duration1, duration2;

	start = clock();
	simpleMatrixMulKernel<<<gridSize,blockSize>>>(d_M1,d_M2,d_MR,n);
	duration1 = (clock()-start) / (double) CLOCKS_PER_SEC;
	start = clock();
	MatrixMulKernel<<<gridSize,blockSize>>>(d_M1,d_M2,d_MR,n);
	duration2 = (clock()-start) / (double) CLOCKS_PER_SEC;
	cudaMemcpy(MR,d_MR,n*n*sizeof(num),cudaMemcpyDeviceToHost);

	cout << "MAtrix MR:" << '\n';
	printMatrix(MR,n);

	cudaFree(d_M1);
	cudaFree(d_M2);
	cudaFree(d_MR);
	cout << "Time1: " <<duration1<<" Time2: "<<duration2 << '\n';
	return 0;
}
