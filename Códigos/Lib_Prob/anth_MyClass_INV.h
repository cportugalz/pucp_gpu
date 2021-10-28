#include <iostream>
#include <complex>
#include <eigen3/Eigen/Eigenvalues> 

#ifndef __MYCLASSINV_H
#define __MYCLASSINV_H

class MiClaseINV {
 private:
	// Definir constantes
	std::complex<double> I = std::complex<double>(0, 1) ;
	double hbar = 6.58211928*1.e-25;
	double clight = 299792458;
	//double GevkmToevsq = hbar*clight*1.e15;
	double GevkmToevsq = 0.197327; // Valor aproximado
  // Definir Matrices para cálculos
  std::complex<double> U[3][3];
  std::complex<double> DM[3][3];

 public:
  MiClaseINV(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double alpha[3], double P[3][3])
	{	
  Eigen::MatrixXcd UPMNS(3, 3);
  Eigen::MatrixXcd Hd(3, 3);
  Eigen::MatrixXcd Pot(3, 3);
  Eigen::MatrixXcd Hff(3, 3);
  Eigen::MatrixXcd S(3, 3);
  Eigen::MatrixXcd V(3, 3);

	double En = Energ*1e9;
	double th12 = th[0];
	double th13 = th[1];
	double th23 = th[2];
	double delta = sigN*dCP;
	double dm21 = dm[0];
	double dm31 = dm[1];
	double alpha1 = alpha[0];
	double alpha2 = alpha[1];
	double alpha3 = alpha[2];
	double rho = sigN*rhomat;

 /* Compute vacuum mixing matrix */
  U[0][0] = cos(th12)*cos(th13);
  U[0][1] = sin(th12)*cos(th13);
  U[0][2] = sin(th13) * exp(-I*delta);

  U[1][0] = -sin(th12)*cos(th23) - cos(th12)*sin(th23)*sin(th13) * exp(I*delta);
  U[1][1] =  cos(th12)*cos(th23) - sin(th12)*sin(th23)*sin(th13) * exp(I*delta);
  U[1][2] =  sin(th23)*cos(th13);

  U[2][0] =  sin(th12)*sin(th23) - cos(th12)*cos(th23)*sin(th13) * exp(I*delta);
  U[2][1] = -cos(th12)*sin(th23) - sin(th12)*cos(th23)*sin(th13) * exp(I*delta);
  U[2][2] =  cos(th23)*cos(th13);

  /* Matriz de masas y Decay */
  DM[0][0] = std::complex<double>(0, -0.5*alpha1/En);
  DM[0][1] = std::complex<double>(0, 0);
  DM[0][2] = std::complex<double>(0, 0);

  DM[1][0] = std::complex<double>(0, 0);
  DM[1][1] = std::complex<double>(0.5*dm21/En, -0.5*alpha2/En);
  DM[1][2] = std::complex<double>(0, 0);

  DM[2][0] = std::complex<double>(0, 0);
  DM[2][1] = std::complex<double>(0, 0);
  DM[2][2] = std::complex<double>(0.5*dm31/En, -0.5*alpha3/En);

  /* Inicializando las matrices para Eigen */
  UPMNS <<
    U[0][0], U[0][1], U[0][2],
    U[1][0], U[1][1], U[1][2], 
    U[2][0], U[2][1], U[2][2];
  Hd <<
    DM[0][0], DM[0][1], DM[0][2],
    DM[1][0], DM[1][1], DM[1][2], 
    DM[2][0], DM[2][1], DM[2][2];
  Pot <<
    std::complex<double>(rho*7.63247*0.5*1.e-14, 0), DM[0][1], DM[0][2],
    DM[1][0], DM[0][0], DM[1][2], 
    DM[2][0], DM[2][1], DM[0][0];

  /* Hamiltoniano final efectivo */
  Hff = UPMNS * Hd * UPMNS.adjoint() + Pot ;

  /* Calculando los autovalores y autovectores */
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> tmp;
  tmp.compute(Hff);

  /* Calculamos la matriz S y ordenamos los autovalores */
  V = tmp.eigenvectors() ;
  S <<
    exp(-I*tmp.eigenvalues()[0]*L*1.e9/GevkmToevsq), DM[0][0], DM[0][0],
    DM[0][0], exp(-I*tmp.eigenvalues()[1]*L*1.e9/GevkmToevsq), DM[0][0], 
    DM[0][0], DM[0][0], exp(-I*tmp.eigenvalues()[2]*L*1.e9/GevkmToevsq);

  S = ( V ) * S * ( V.inverse() );

  // Calculando la matriz de probabilidad
  for(int i=0; i<3; i++){
  for(int j=0; j<3; j++){
  P[i][j] = abs(S.col(i)[j]*S.col(i)[j]); } } // Fila j , columna i

  } 
  


};

#endif
