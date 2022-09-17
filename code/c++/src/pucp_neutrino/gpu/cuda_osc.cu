#include "probabilities.h"
#include <cuda_runtime.h>


// GPU Standard Oscilation
void cuda_StandardOscilation(
	cuDoubleComplex* _U, int _size_data, int _sigN, double _L, double _rho,
	double* _dm, double* _alpha){
		cuda_InvisibleDecay(_U, _size_data, _sigN, _L, _rho, _dm, _alpha);
}
