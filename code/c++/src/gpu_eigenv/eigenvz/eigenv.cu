#include "eigenv.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>


__global__ void eigenv(int _batch, cuDoubleComplex* _dinput){
    int tidx = blockDim.x*blockIdx.x + threadIdx.x;
    if (tidx < _batch){
        printf("%e ", _dinput[tidx].x);
    }

}

void eigen_system(int _batch, cuDoubleComplex* _input, double _error){
    using data_type =  cuDoubleComplex;
    data_type* dinput = nullptr;
    // data_type* dbatched[_batch];
    cudaMalloc(reinterpret_cast<void **>(&dinput), 9*sizeof(data_type));
    cudaMemcpy(dinput, _input, 9*sizeof(data_type), cudaMemcpyHostToDevice);
    float threads = 1024;
    int blocks = ceil(_batch/threads);
    eigenv<<< blocks, threads>>>(_batch, dinput);
    cudaFree(dinput);
}
