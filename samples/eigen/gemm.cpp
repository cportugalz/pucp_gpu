#include <eigen3/Eigen/Eigen>
#include <complex>
#include <cstdio>

void print_matrix(Eigen::Matrix3cd _matrix) {
	for (int i=0; i< 3; i++) {
		for( int j=0; j<3; j++){
			printf("%e %e\t", _matrix(i,j).real(), _matrix(i,j).imag());
		}
		printf("\n");
	}
}

int main(){
	Eigen::Matrix3cd A;
	// A << std::complex<double>(8.216101e-01, 0.0) , std::complex<double>(5.501137e-01, 0.0) , 
	// std::complex<double>(1.190016e-04, 1.494381e-01) , -std::complex<double>(3.714239e-01, 9.246545e-02) , 
	// std::complex<double>(5.545727e-01, 6.191077e-02) , std::complex<double>(7.362816e-01, 0.0) ,
	// std::complex<double>(4.142244e-01, 8.288165e-02) , std::complex<double>(-6.187985e-01, 5.549388e-02) , 
	// std::complex<double>(6.599679e-01 + 0.0);
	A << std::complex<double>(2,0), std::complex<double>(0,-1), std::complex<double>(1,0),
	std::complex<double>(0,3),std::complex<double>(5,0), std::complex<double>(0,2),
	std::complex<double>(4,0), std::complex<double>(0,9), std::complex<double>(6,0); 
	Eigen::Matrix3cd B;
	// B << std::complex<double>(0.0, -0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0) ,
	// std::complex<double>(0.0, 0.0), std::complex<double>(3.7e-12, 0.0), std::complex<double>(0.0, 0.0),
	// std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(1.25e-10, -0.0);
	B << std::complex<double>(1,0), std::complex<double>(0,4), std::complex<double>(0,-2),
	std::complex<double>(0,7), std::complex<double>(2,0), std::complex<double>(3,0),
	std::complex<double>(0,5), std::complex<double>(1,0), std::complex<double>(5,0);
	Eigen::Matrix3cd C;
	C = A*B;
	print_matrix(A);
	print_matrix(B);
	print_matrix(C);

	return 0;
}