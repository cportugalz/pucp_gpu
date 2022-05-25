#include "probabilities.h"
#include "utils.h"
#include <iostream>


void event_total(double** _plusPrSTD, double** _minusPrSTD) {
    double signal1 = 1002.7*_plusPrSTD[1][0] + 25.1;
    double signal2 = 418.4*_minusPrSTD[1][0] + 14.6;
    double background1 = 24.6*_plusPrSTD[1][1] + 6;
    double background2 = 15*_minusPrSTD[1][1] + 4;
    double ntotal = signal1 + signal2 + background1 + background2;
    std::cout<<ntotal<<std::endl;
}


int main() {
    /* This program calculates the energy for oscillation probability using 
    two signals and two backgrounds
    **/
    double L = 1300;
    // 
    double GevkmToevsq = 0.197327;
    double th[] = { 0.163179, 0.216797, 0.841411 };
    double delta = 2.10451;
    double dm[] = { 6.18e-4, 2.147e-3 }; 
    double gamma2 = 4e-24;

    // additional variables
    double alpSTD[3] = { 0 };
    double rho = 2.956740;
    // for standard oscilation, invisible decay, vpd and nsi
	std::complex<double>** U1_plusone = make_umns(1, th, delta);
    std::complex<double>** U1_minusone = make_umns(-1, th, delta);
    
    double** plusPrSTD = new double*[3];
    double** minusPrSTD = new double*[3];
    for ( int i=0; i<3; ++i){
		plusPrSTD[i] = new double[3];
		minusPrSTD[i] = new double[3];
	}
    StandardOscilation(
		U1_plusone, 0.5, 1, L, rho, dm, alpSTD, plusPrSTD
	);
     StandardOscilation(
		U1_minusone, 0.5, -1, L, rho, dm, alpSTD, minusPrSTD
	);
    event_total(plusPrSTD, minusPrSTD);
    
    return 0;
}