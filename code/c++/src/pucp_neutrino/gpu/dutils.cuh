#pragma once
#ifndef _DUTILS_CUH
#define _DUTILS_CUH
#include <cuComplex.h>

__device__ inline cuDoubleComplex exp(cuDoubleComplex _num){
	return make_cuDoubleComplex(exp(_num.x)*cos(_num.y), exp(_num.x)*sin(_num.y));
}

#endif