#ifndef __CUDA_SIMULATION_H
#define __CUDA_SIMULATION_H


void cuda_simulation_StandardOscilation(
	int num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho, 
	double* _dm, double* _alpha);


void cuda_simulation_InvisibleDecay(
	int num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho, 
	double* _dm, double* _alpha);


void cuda_simulation_NonStandardInteraction(
	int num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho, 
	double* _dm, double* _alpha);


void cuda_simulation_ViolationEquivalence(
	int num_simulations, int _sg, double* _th, double _dcp,  double _L, double _rho, 
	double* _dm, double* _alpha);
#endif