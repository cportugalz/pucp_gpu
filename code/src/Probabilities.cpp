#include "Probabilities.h"
#include <eigen3/Eigen/Eigenvalues>
#include <complex>

const std::complex<double> ProbConst::I = std::complex<double>(0, 1);
const double ProbConst::hbar = 6.58211928*1.e-25;
const double ProbConst::clight = 299792458;
//double GevkmToevsq = hbar*clight*1.e15;
const double ProbConst::GevkmToevsq = 0.197327; // Valor aproximado

Probabilities::Probabilities() {

}

Probabilities::~Probabilities() {

}

void Probabilities::StandardOscilation(
	double Energ, int sigN, double L, double rhomat, double* th, 
    double dCP, double* dm, double* alpha, double** P) {
	InvisibleDecay(Energ, sigN, L, rhomat, th, dCP, dm, alpha, P);	
}

void Probabilities::InvisibleDecay(
    double Energ, int sigN, double L, double rhomat, double* th, 
    double dCP, double* dm, double* alpha, double** P) {
	Eigen::MatrixXcd UPMNS(3, 3);
	Eigen::MatrixXcd Hd(3, 3);
	Eigen::MatrixXcd Pot(3, 3);
	Eigen::MatrixXcd Hff(3, 3);
	Eigen::MatrixXcd S(3, 3);
	Eigen::MatrixXcd V(3, 3);

	double En = Energ * 1e9;
	double th12 = th[0];
	double th13 = th[1];
	double th23 = th[2];
	double delta = sigN * dCP;
	double dm21 = dm[0];
	double dm31 = dm[1];
	double alpha1 = alpha[0];
	double alpha2 = alpha[1];
	double alpha3 = alpha[2];
	double rho = sigN * rhomat;
	// Definir Matrices para cálculos
    std::complex<double> U[3][3];
    std::complex<double> DM[3][3];

	/* Compute vacuum mixing matrix */
	U[0][0] = cos(th12) * cos(th13);
	U[0][1] = sin(th12) * cos(th13);
	U[0][2] = sin(th13) * exp(-ProbConst::I * delta);

	U[1][0] = -sin(th12) * cos(th23) - cos(th12) * sin(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[1][1] = cos(th12) * cos(th23) - sin(th12) * sin(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[1][2] = sin(th23) * cos(th13);

	U[2][0] = sin(th12) * sin(th23) - cos(th12) * cos(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[2][1] = -cos(th12) * sin(th23) - sin(th12) * cos(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[2][2] = cos(th23) * cos(th13);

	/* Matriz de masas y Decay */
	DM[0][0] = std::complex<double>(0, -0.5 * alpha1 / En);
	DM[0][1] = std::complex<double>(0, 0);
	DM[0][2] = std::complex<double>(0, 0);

	DM[1][0] = std::complex<double>(0, 0);
	DM[1][1] = std::complex<double>(0.5 * dm21 / En, -0.5 * alpha2 / En);
	DM[1][2] = std::complex<double>(0, 0);

	DM[2][0] = std::complex<double>(0, 0);
	DM[2][1] = std::complex<double>(0, 0);
	DM[2][2] = std::complex<double>(0.5 * dm31 / En, -0.5 * alpha3 / En);

	/* Inicializando las matrices para Eigen */
	UPMNS << U[0][0], U[0][1], U[0][2],
		U[1][0], U[1][1], U[1][2],
		U[2][0], U[2][1], U[2][2];
	Hd << DM[0][0], DM[0][1], DM[0][2],
		DM[1][0], DM[1][1], DM[1][2],
		DM[2][0], DM[2][1], DM[2][2];
	Pot << std::complex<double>(rho * 7.63247 * 0.5 * 1.e-14, 0), DM[0][1], DM[0][2],
		DM[1][0], DM[0][0], DM[1][2],
		DM[2][0], DM[2][1], DM[0][0];

	/* Hamiltoniano final efectivo */
	Hff = UPMNS * Hd * UPMNS.adjoint() + Pot;

	/* Calculando los autovalores y autovectores */
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> tmp;
	tmp.compute(Hff);

	/* Calculamos la matriz S y ordenamos los autovalores */
	V = tmp.eigenvectors();
	S << exp(-ProbConst::I * tmp.eigenvalues()[0] * L * 1.e9 / ProbConst::GevkmToevsq), DM[0][0], DM[0][0],
		DM[0][0], exp(-ProbConst::I * tmp.eigenvalues()[1] * L * 1.e9 / ProbConst::GevkmToevsq), DM[0][0],
		DM[0][0], DM[0][0], exp(-ProbConst::I * tmp.eigenvalues()[2] * L * 1.e9 / ProbConst::GevkmToevsq);

	S = (V)*S * (V.inverse());

	// Calculando la matriz de probabilidad
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			P[i][j] = abs(S.col(i)[j] * S.col(i)[j]);
		}
	} // Fila j , columna i
}

void Probabilities::NonStandardInteraction(
    double Energ, int sigN, double L, double rhomat, double* th, 
    double dCP, double* dm, double* parmNSI, double** P) {
    Eigen::MatrixXcd UPMNS(3, 3);
    Eigen::MatrixXcd Hd(3, 3);
    Eigen::MatrixXcd Pot(3, 3);
    Eigen::MatrixXcd MNSI(3, 3);
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
	double rho = sigN*rhomat;
    // Definir Matrices para cálculos
    std::complex<double> U[3][3];
    std::complex<double> DM[3][3];
    std::complex<double> NSI[3][3];

    /* Compute vacuum mixing matrix */
    U[0][0] = cos(th12)*cos(th13);
    U[0][1] = sin(th12)*cos(th13);
    U[0][2] = sin(th13) * exp(-ProbConst::I*delta);

    U[1][0] = -sin(th12)*cos(th23) - cos(th12)*sin(th23)*sin(th13) * exp(ProbConst::I*delta);
    U[1][1] =  cos(th12)*cos(th23) - sin(th12)*sin(th23)*sin(th13) * exp(ProbConst::I*delta);
    U[1][2] =  sin(th23)*cos(th13);

    U[2][0] =  sin(th12)*sin(th23) - cos(th12)*cos(th23)*sin(th13) * exp(ProbConst::I*delta);
    U[2][1] = -cos(th12)*sin(th23) - sin(th12)*cos(th23)*sin(th13) * exp(ProbConst::I*delta);
    U[2][2] =  cos(th23)*cos(th13);

    /* Matriz de masas */
    DM[0][0] = std::complex<double>(0, 0);
    DM[0][1] = std::complex<double>(0, 0);
    DM[0][2] = std::complex<double>(0, 0);

    DM[1][0] = std::complex<double>(0, 0);
    DM[1][1] = std::complex<double>(0.5*dm21/En, 0);
    DM[1][2] = std::complex<double>(0, 0);

    DM[2][0] = std::complex<double>(0, 0);
    DM[2][1] = std::complex<double>(0, 0);
    DM[2][2] = std::complex<double>(0.5*dm31/En, 0);
    
    /* Matriz de parámetros NSI */
    NSI[0][0] = parmNSI[0];
    NSI[0][1] = parmNSI[3] * exp(ProbConst::I*parmNSI[4]);
    NSI[0][2] = parmNSI[5] * exp(ProbConst::I*parmNSI[6]);

    NSI[1][0] = parmNSI[3] * exp(-ProbConst::I*parmNSI[4]);
    NSI[1][1] = parmNSI[1];
    NSI[1][2] = parmNSI[7] * exp(ProbConst::I*parmNSI[8]);

    NSI[2][0] = parmNSI[5] * exp(-ProbConst::I*parmNSI[6]);
    NSI[2][1] = parmNSI[7] * exp(-ProbConst::I*parmNSI[8]);
    NSI[2][2] = parmNSI[2];

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
	MNSI <<
		NSI[0][0], NSI[0][1], NSI[0][2],
		NSI[1][0], NSI[1][1], NSI[1][2], 
		NSI[2][0], NSI[2][1], NSI[2][2];

	/* Hamiltoniano final efectivo */
	Hff = UPMNS * Hd * UPMNS.adjoint() + rho*7.63247*0.5*1.e-14*MNSI + Pot ;

	/* Calculando los autovalores y autovectores */
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> tmp;
	tmp.compute(Hff);

	/* Calculamos la matriz S y ordenamos los autovalores */
	V = tmp.eigenvectors() ;
	S <<
		exp(-ProbConst::I*tmp.eigenvalues()[0]*L*1.e9/ProbConst::GevkmToevsq), DM[0][0], DM[0][0],
		DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[1]*L*1.e9/ProbConst::GevkmToevsq), DM[0][0], 
		DM[0][0], DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[2]*L*1.e9/ProbConst::GevkmToevsq);

	S = ( V ) * S * ( V.inverse() );

	// Calculando la matriz de probabilidad
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++) {
			P[i][j] = abs(S.col(i)[j]*S.col(i)[j]); 
		} 
	} // Fila j , columna i
}

void Probabilities::ViolationPrincipleDecay(
	double Energ, int sigN, double L, double rhomat, double* th, 
	double dCP, double* dm, double* gamma, double** P) {
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
	double gamma1 = gamma[0];
	double gamma2 = gamma[1];
	double gamma3 = gamma[2];
	double rho = sigN*rhomat;
	std::complex<double> U[3][3];
    std::complex<double> DM[3][3];
	/* Compute vacuum mixing matrix */
	U[0][0] = cos(th12)*cos(th13);
	U[0][1] = sin(th12)*cos(th13);
	U[0][2] = sin(th13) * exp(-ProbConst::I*delta);

	U[1][0] = -sin(th12)*cos(th23) - cos(th12)*sin(th23)*sin(th13) * exp(ProbConst::I*delta);
	U[1][1] =  cos(th12)*cos(th23) - sin(th12)*sin(th23)*sin(th13) * exp(ProbConst::I*delta);
	U[1][2] =  sin(th23)*cos(th13);

	U[2][0] =  sin(th12)*sin(th23) - cos(th12)*cos(th23)*sin(th13) * exp(ProbConst::I*delta);
	U[2][1] = -cos(th12)*sin(th23) - sin(th12)*cos(th23)*sin(th13) * exp(ProbConst::I*delta);
	U[2][2] =  cos(th23)*cos(th13);

	/* Matriz de masas y VEP con U_g = U */
	DM[0][0] = std::complex<double>(0+2*En*gamma1, 0);
	DM[0][1] = std::complex<double>(0, 0);
	DM[0][2] = std::complex<double>(0, 0);

	DM[1][0] = std::complex<double>(0, 0);
	DM[1][1] = std::complex<double>(0.5*dm21/En + 2*En*gamma2, 0);
	DM[1][2] = std::complex<double>(0, 0);

	DM[2][0] = std::complex<double>(0, 0);
	DM[2][1] = std::complex<double>(0, 0);
	DM[2][2] = std::complex<double>(0.5*dm31/En + 2*En*gamma3, 0);

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
		exp(-ProbConst::I*tmp.eigenvalues()[0]*L*1.e9/ProbConst::GevkmToevsq), DM[0][0], DM[0][0],
		DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[1]*L*1.e9/ProbConst::GevkmToevsq), DM[0][0], 
		DM[0][0], DM[0][0], exp(-ProbConst::I*tmp.eigenvalues()[2]*L*1.e9/ProbConst::GevkmToevsq);

	S = ( V ) * S * ( V.inverse() );

	// Calculando la matriz de probabilidad
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) {
			P[i][j] = abs(S.col(i)[j]*S.col(i)[j]);
		}
	}
}

