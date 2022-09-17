#ifndef PROBABILITIES_H
#define PROBABILITIES_H

#include <cuComplex.h>


// GPU Standard Oscilation
void cuda_StandardOscilation(
	cuDoubleComplex* _U, int _size_data, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha);


// GPU Invisible Decay
void cuda_InvisibleDecay(
	cuDoubleComplex* _U, int _size_data, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha);


// GPU Non Standard Interaction
void cuda_NonStandardInteraction(
	cuDoubleComplex* _U, int _size_data, int _sigN, double _L, double _rho, 
	double* _dm, double* _alpha);


#endif