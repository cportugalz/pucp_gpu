#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <sys/resource.h>
#include "eigenvz/eigenv.h"
#include <iostream>


int main(int argc, char* argv[]){
    using data_type =  cuDoubleComplex;
    printf("Initiating Eigen Values/Vector on GPU\n");
    double error = 1e-8; // Límite de error para convergencia
	int i,j, size=3;
	struct rlimit rl{1<<28, 1l<<32};
	setrlimit(RLIMIT_STACK, &rl);
	printf("Stack size: %lu MiB up to %lu GiB\n", rl.rlim_cur/(1<<20), rl.rlim_max/(1<<30));
	if(argc < 2){
		std::cout << "Usage: make n=<your number of matrices>" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::string sim_arg(argv[1]);
	int num_operations = std::stod(sim_arg);
    std::complex<double> A[size][size]; 
    std::complex<double> I = std::complex<double>(0,1);
   

    // Matriz de Prueba
    /*A[0][0] = 0.85 + 0.62*I; A[0][1] = 0.91 - 0.03*I; A[0][2] = 0.61;
    A[1][0] = 0.065; A[1][1] = 0.51 + 0.98*I; A[1][2] = 0.1 + I;
    A[2][0] = 0.78 - 0.002*I; A[2][1] = 0.951 - 0.369*I; A[2][2] = 0.55 + 0.72*I;*/

    // Matriz de Prueba
    /*A[0][0] = 8.5 + 6.2*I; A[0][1] = 9.1 - 3*I; A[0][2] = 6.1;
    A[1][0] = 6.5; A[1][1] = 5.1 ; A[1][2] = 1 + I;
    A[2][0] = 7.8 - 2*I; A[2][1] = 9.51 - 3.69*I; A[2][2] = 5.5 + 7.2*I;*/

    // Matriz de Prueba
    A[0][0] = 0.1-0.9*I; A[0][1] = 0.8-0.7*I; A[0][2] = 0.9+0.68*I;
	A[1][0] = 0.7-0.23*I; A[1][1] = 0.5 + 0.12*I ; A[1][2] = 0.9 + 0.5*I ;
	A[2][0] = 0.55+0.4*I; A[2][1] = -0.43+0.8*I; A[2][2] = 0.2-0.9*I;

    data_type* input = new data_type[9];
	printf("Input:\n");
	for(int i=0; i<size; i++){
		for(int j=0; j<size;j++){
			// printf("%f %f\t", U[i][j].real(), U[i][j].imag());
			input[i*size+j] = make_cuDoubleComplex(A[i][j].real(), A[i][j].imag());
			printf("Input[%d][%d]:(%e %e)\t ", i,j,input[i*size+j].x, input[i*size+j].y);
		}
		printf("\n");
	}
    eigen_system(num_operations, input, error);
    delete input;
    return 0;
}