#include <iostream>
#include <fstream>
#include <complex>
#include <eigen3/Eigen/Eigenvalues> 
#include <libIntegrate/_1D/GaussianQuadratures/GaussLegendre.hpp>
#include <functional>
#include <iomanip>
#include <vector>
#include <cmath>
using namespace std;

#ifndef __MYCLASSVIS_H
#define __MYCLASSVIS_H
 

class MiClaseVIS {
 private:
  // Matriz Potencial
  Eigen::Matrix3cd Pot;//(3, 3)
  // Matrices UPMNS 
  Eigen::Matrix3cd Utemp;//(3,3)
  // Matrices temp Hamilt y Pot 
  Eigen::Matrix3cd Htemp;//(3, 3)
  // Objeto para almacenar el Eigensistema 
  Eigen::ComplexEigenSolver<Eigen::Matrix3cd> tmpVD;

	// Definir constantes
	std::complex<double> I = std::complex<double>(0, 1) ;
	std::complex<double> Z0 = std::complex<double>(0, 0) ;
	double th12, th13, th23, delta;
	//double hbar = 6.58211928*1.e-25;
	//double clight = 299792458;
	//double GevkmToevsq = hbar*clight*1.e15;
	double GevkmToevsq = 0.197327;
	double L; /* Longitud */

	int fi /* flavor inicial */, si;
	int ff /* flavor final */, sf;
	int par /* padre */, hij /* hijo */;
	int qcoup /* acoplamiento */;
	double Enf; /* Energía final */
    	double mm; /* variable temporal */
	double mss[3];
	double alphavac[3];


  /* Algoritmo para Integral */
  _1D::GQ::GaussLegendreQuadrature<double,64> NintegrGLQ; // 64

  double ex2(double x)
  {return x*x ;}

  double Min(double x, double y)
  { if (x<y)
    return x;
    else
    return y; }
  
  // Matriz U PMNS
  int MUPMNS(int sg, double th12, double th13, double th23, double dcp)
  {
  double delta = sg*dcp;
  std::complex<double> U[3][3];
  U[0][0] = cos(th12)*cos(th13);
  U[0][1] = sin(th12)*cos(th13);
  U[0][2] = sin(th13) * exp(-I*delta);

  U[1][0] = -sin(th12)*cos(th23) - cos(th12)*sin(th23)*sin(th13) * exp(I*delta);
  U[1][1] =  cos(th12)*cos(th23) - sin(th12)*sin(th23)*sin(th13) * exp(I*delta);
  U[1][2] =  sin(th23)*cos(th13);

  U[2][0] =  sin(th12)*sin(th23) - cos(th12)*cos(th23)*sin(th13) * exp(I*delta);
  U[2][1] = -cos(th12)*sin(th23) - sin(th12)*cos(th23)*sin(th13) * exp(I*delta);
  U[2][2] =  cos(th23)*cos(th13);

  Utemp <<
    U[0][0], U[0][1], U[0][2],
    U[1][0], U[1][1], U[1][2], 
    U[2][0], U[2][1], U[2][2];

  return 0;}
  
  // Matriz H: hamiltoniano
  int HamilDc(int q, double Eni)
  {
  MUPMNS(q,th12,th13,th23,delta);
  Htemp <<
    std::complex<double>(0.5*mss[0]/Eni, -0.5*alphavac[0]/Eni), Z0, Z0,
    Z0, std::complex<double>(0.5*mss[1]/Eni, -0.5*alphavac[1]/Eni), Z0, 
    Z0, Z0, std::complex<double>(0.5*mss[2]/Eni, -0.5*alphavac[2]/Eni);

  /* Hamiltoniano */
  Htemp = Utemp * Htemp * Utemp.adjoint() + q*Pot ;

  return 0;
  }


  double dGdE(double mi, double mf, double Ei, double Ef, int coup)
  {
  double xif = mi/mf, ei=Ei*1.e-9, ef=Ef*1.e-9, tmp;
  if(coup==1)
  {if(ei/xif <= ef && ef <= ei)
  {tmp = (1./sqrt(1-mi/ex2(ei*1.e9)))*(xif/(xif-1))*(1./(ei*ef))*(ex2(ei + sqrt(xif)*ef)/ex2(sqrt(xif)+1));}
  else {tmp=0;}
  }
  else
  {if(ei/xif <= ef && ef <= ei)
  {tmp = (1./sqrt(1-mi/ex2(ei*1.e9)))*(xif/(xif-1))*(1./(ei*ef))*(ex2(ei - sqrt(xif)*ef)/ex2(sqrt(xif)-1));}
  else {tmp=0;}
  }
  return tmp;}

  double dGbdE(double mi, double mf, double Ei, double Ef, int coup)
  {
  double xif = mi/mf, ei=Ei*1.e-9, ef=Ef*1.e-9, tmp;
  if(coup==1)
  {if(ei/xif <= ef && ef <= ei)
  {tmp = (1./sqrt(1-mi/ex2(ei*1.e9)))*(xif/(xif-1))*((ei-ef)/(ei*ef))*((xif*ef-ei)/ex2(sqrt(xif)+1));}
  else {tmp=0;}
  }
  else
  {if(ei/xif <= ef && ef <= ei)
  {tmp = (1./sqrt(1-mi/ex2(ei*1.e9)))*(xif/(xif-1))*((ei-ef)/(ei*ef))*((xif*ef-ei)/ex2(sqrt(xif)-1));}
  else {tmp=0;}
  }
  return tmp;}
  
  double IntegrandoPhi(double ei)
  {
  double Eni = ei*1.e9;
  double ef = Enf*1.e-9;
  double dist = L*1.e9/GevkmToevsq;
  /* Calculamos Autovalores y Autovectores iniiales */
  HamilDc(si,Eni);
  tmpVD.compute(Htemp);
  
  /* Masas y alpha tíldes iniciales */
  double massfisq[3]={2*Eni*real(tmpVD.eigenvalues()[0]),2*Eni*real(tmpVD.eigenvalues()[1]),2*Eni*real(tmpVD.eigenvalues()[2])};
  double alphafi[3]={-2*Eni*imag(tmpVD.eigenvalues()[0]),-2*Eni*imag(tmpVD.eigenvalues()[1]),-2*Eni*imag(tmpVD.eigenvalues()[2])};

  
  /* Matriz de autovectores iniciales */
  Eigen::MatrixXcd Umati(3, 3);
  Eigen::MatrixXcd Umatinvi(3, 3);
  Umati = tmpVD.eigenvectors();
  Umatinvi = Umati.inverse();
 
  /* Matrices Cmat inicial */
  Eigen::MatrixXcd Cmati(3, 3);
  Cmati = (Umati.transpose())*(Utemp.conjugate());

  /* Calculamos Autovalores y Autovectores finales */
  HamilDc(sf,Enf);
  tmpVD.compute(Htemp);
  
  
  /* Masas y alpha tíldes finales */
  double massffsq[3]={2*Enf*real(tmpVD.eigenvalues()[0]),2*Enf*real(tmpVD.eigenvalues()[1]),2*Enf*real(tmpVD.eigenvalues()[2])};
  double alphaff[3]={-2*Enf*imag(tmpVD.eigenvalues()[0]),-2*Enf*imag(tmpVD.eigenvalues()[1]),-2*Enf*imag(tmpVD.eigenvalues()[2])};
  
  /* Matriz de autovectores finales */
  Eigen::MatrixXcd Umatf(3, 3);
  Umatf = tmpVD.eigenvectors();

  /* Matrices Cmat final */
  Eigen::MatrixXcd Cmatinvf(3, 3);
  Cmatinvf = (Umatf.transpose())*(Utemp.conjugate());
  Cmatinvf = Cmatinvf.inverse();


  /* Funciones de cambio de quiralidad */
  double Theta=0;
  if(si*sf>0)
  {Theta = dGdE(mss[par],mss[hij],Eni,Enf,qcoup);}
  else
  {Theta = dGbdE(mss[par],mss[hij],Eni,Enf,qcoup);}

  /* Suma total */
  double sum=0, tmp=0, txxx = 0, txxx2=0;
  int i,p,h,n;


  for(i=0; i<3; i++)
  {for(p=0; p<3; p++)
  {for(h=0; h<3; h++)
  {for(n=0; n<3; n++)
  {

  sum = sum + real( Umatinvi.col(fi)[i]*conj(Umatinvi.col(fi)[p]) *Umatf.col(h)[ff]*conj(Umatf.col(n)[ff]) * (((ef/ei)*(alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n]) - I*((ef/ei)*((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n]))))/(ex2((ef/ei)*(alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n])) + ex2((ef/ei)*((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n]))))) * ( exp(-I*((massffsq[h]) - (massffsq[n]))/(2*Enf)*dist)*exp(-((alphaff[h] + alphaff[n])/(2*Enf))*dist) - exp(-I*((massfisq[i]) - (massfisq[p]))/(2*Eni)*dist) * exp(-((alphafi[i] + alphafi[p])/(2*Eni))*dist) ) * Cmatinvf.col(h)[hij]*conj(Cmatinvf.col(n)[hij])*Cmati.col(par)[i]*conj(Cmati.col(par)[p]) );


  }
  }
  }
  }
  

  tmp = 2*sum*(((ef/ei)*alphavac[par])/Eni)*Theta;

  return tmp;}


 public:
  MiClaseVIS(double E, double Long, double rho, double th[3], double dCP, double dm[2], double mlight /* masa light */, double alpha[3], int tfi /* flavor inicial */, int tsi /* signo inicial */, int tff /* flavor final */, int tsf /* signo final */, int tpar /* padre */, int thij /* hijo */, int tqcoup /* acoplamiento */, double P[1] /* Probabilidad */)
  {	
	th12 = th[0]; th13 = th[1]; th23 = th[2]; delta = dCP; 
	/* definimos vector mss */
	if (dm[1] > 0)
	{mss[0] = mlight; mss[1] = dm[0] + mlight; mss[2] = dm[1] + mlight; }
	else { mss[0] = mlight; mss[1] = -dm[1] + mlight; mss[2] = dm[0] - dm[1] + mlight ; } 
	alphavac[0] = alpha[0]; alphavac[1] = alpha[1]; alphavac[2] = alpha[2];
	L = Long;
	
  /* Inicializamos el potencial */
  Pot <<
    std::complex<double>(rho*7.63247*0.5*1.e-14, 0), Z0, Z0,
    Z0, Z0, Z0, 
    Z0, Z0, Z0;

  /* Esta linea de código es necesaria para que funcione la función NintegrGLQ */
  using std::placeholders::_1;

  // Iniciamos los parámetros de sabores, signos, padre, hijo y acoplamientos
  fi= tfi; si= tsi; ff= tff; sf= tsf; par= tpar; hij= thij; qcoup= tqcoup ;

  Enf=E*1.e9; // Energia final
  mm = Min(20 , (mss[par]/mss[hij])*E ); // Límite superior de integración
  
  // Calculamos la probabilidad
  
  //printf("\n masa1: %g , mas2: %g , mas3: %g \n", sqrt(mss[0]), sqrt(mss[1]), sqrt(mss[2]) );
  //printf("\n IntegrandoPhi: %g \n", IntegrandoPhi(1.9) );
  //printf("\n IntegrandoPhi: %g \n", IntegrandoPhi(2.9) );
  //printf("\n IntegrandoPhi: %g \n", IntegrandoPhi(5.9) );
  
  P[0] = 1.e9*NintegrGLQ(std::bind(&MiClaseVIS::IntegrandoPhi, this, _1 ), (1.00001)*E, mm);
  //printf("\n E: %g , Prob = %g (despues de integracion)", E, P[2][2]);
  }

};

#endif
