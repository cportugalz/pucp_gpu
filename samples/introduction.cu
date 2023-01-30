#include <stdio.h>
#include <cuda_runtime.h>


__global__ void test(int **a, int **b, int **c, int n){
	int tidx =  blockDim.x * blockIdx.x + threadIdx.x;
	int tidy =  blockDim.y * blockIdx.y + threadIdx.y;
	int tidz =  blockDim.z * blockIdx.z + threadIdx.z;
	
	if (tidx < n){
		printf("ThreadIdx: %d ThreadIdy: %d ThreadIdz: %d\n", tidx, tidy, tidz);
		//printf("ThreadIdx: %d\n", tidx);
		/*for (int i=0; i< n; i++){
			c[tidx][i] = a[tidx][i] + b[tidx][i];
		}*/
		c[tidx][tidy*2+ tidz] = a[tidx][tidy*2+tidz] + b[tidx][tidy*2+tidz];
	}
}

int main(){
	int ha[][4] = { {1,2,3,4}, {1,2,3,4}, {1,2,3,4}, {1,2,3,4}};
	int hb[][4] = { {1,6,4,1}, {2,2,3,34}, {1,234,3,12}, {1,2,4,6}};
	int **da;
	int **db;
	int **dc;
	int numElements = 4;
	int **hc = (int**)malloc(numElements* sizeof(int*));
	for (int i=0; i <numElements; i++){
		hc[i] = (int*)malloc(numElements* sizeof(int));
	}
	cudaMalloc((void**)&da, sizeof(int*) * numElements );
	cudaMalloc((void**)&db, sizeof(int*) * numElements );
	cudaMalloc((void**)&dc, sizeof(int*) * numElements );
	int *pda[numElements], *pdb[numElements], *pdc[numElements];
	for (int i=0; i < numElements; i++){
		cudaMalloc((void**)&pda[i], sizeof(int) * numElements);
		cudaMalloc((void**)&pdb[i], sizeof(int) * numElements);
		cudaMalloc((void**)&pdc[i], sizeof(int) * numElements);
	
	}
	
	for (int i=0; i < numElements; i++){
		cudaMemcpy(pda[i], ha[i], sizeof(int) * numElements, cudaMemcpyHostToDevice);
		cudaMemcpy(pdb[i], hb[i], sizeof(int) * numElements, cudaMemcpyHostToDevice);
	}
	cudaMemcpy(da, pda, sizeof(int*) * numElements, cudaMemcpyHostToDevice);
	cudaMemcpy(db, pdb, sizeof(int*) * numElements, cudaMemcpyHostToDevice);
	cudaMemcpy(dc, pdc, sizeof(int*) * numElements, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(4,2,2);
	test<<<1, threadsPerBlock>>>(da, db, dc, numElements);
	cudaDeviceSynchronize();
	for (int i=0; i <numElements; i++){
		cudaMemcpy(hc[i], pdc[i], sizeof(int)*numElements, cudaMemcpyDeviceToHost);
	}
	
	for (int i=0; i <numElements; i++){
		for (int j=0; j <numElements; j++){
			printf("c[%d][%d]:%d \t", i,j, hc[i][j]);
		}
		printf("\n");
	}
	for (int i=0; i <numElements; i++){
		free(hc[i]);
		cudaFree(pda);
		cudaFree(pdb);
		cudaFree(pdc);
	}
	free(hc);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}