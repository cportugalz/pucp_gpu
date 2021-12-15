#include "Probabilities.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip> // Compatibility  with linux gcc

int main(int argc, char* argv[]){

    if(argc < 2){
        std::cout << "Usage: make single n=<your number>" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string sim_arg(argv[1]);
    int num_simulations = std::stod(sim_arg);
    // Parmámetros para Oscilación Estándar
    double d = -1.57;
    double L = 1300; double rho = 2.956740;
    int s = 1; int a = 1 , b = 0 ;

    double ee = 0 ; double mm = 0; double tt = 0;
    double em = 0.05 ; double emf = -1.55;
    double et = 0 ; double etf = 0;
    double mt = 0 ; double mtf = 0;
    double th[] = { 0.59, 0.15, 0.84 };
    double th12 = th[0];
	double th13 = th[1];
	double th23 = th[2];
    double dm[] = { 7.4e-5, 2.5e-3 }; 
    double alpSTD[3] = { 0 };
    double alpINV[3] = { 0, 0, 5.e-5 };
    double alpVEP[3] = { 0, 0, 2.e-24 };
    double alpNSI[] = { ee, mm, tt, em, emf, et, etf, mt, mtf };
    double delta = s * d;
    // U matrix for calculations
    std::complex<double>** U = new std::complex<double>*[3];
    for (int i=0; i<3; i++){
        U[i] = new std::complex<double>[3];
    }
    U[0][0] = cos(th12) * cos(th13);
	U[0][1] = sin(th12) * cos(th13);
	U[0][2] = sin(th13) * exp(-ProbConst::I * delta);
	U[1][0] = -sin(th12) * cos(th23) - cos(th12) * sin(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[1][1] = cos(th12) * cos(th23) - sin(th12) * sin(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[1][2] = sin(th23) * cos(th13);
	U[2][0] = sin(th12) * sin(th23) - cos(th12) * cos(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[2][1] = -cos(th12) * sin(th23) - sin(th12) * cos(th23) * sin(th13) * exp(ProbConst::I * delta);
	U[2][2] = cos(th23) * cos(th13);


    double** PrSTD = new double*[3];
    double** PrINV = new double*[3];
    double** PrVEP = new double*[3];
    double** PrNSI = new double*[3];
    for ( int i=0; i<3; ++i){
        PrSTD[i] = new double[3];
        PrINV[i] = new double[3];
        PrVEP[i] = new double[3];
        PrNSI[i] = new double[3];
    }


    std::ofstream file_results("output.txt");
    auto start_time = std::chrono::high_resolution_clock::now();
    if(file_results.is_open()){
        num_simulations = num_simulations/100.0;
        for(double ene=0.01 ; ene<=num_simulations; ) {
            StandardOscilation(
                U, ene, s, L, rho, th, dm, alpSTD, PrSTD
            );
            InvisibleDecay(
                U, ene, s, L, rho, th, dm, alpINV, PrINV 
            );
            ViolationPrincipleDecay(
                U, ene, s, L, rho, th, dm, alpVEP, PrVEP 
            );        
            NonStandardInteraction(
                U, ene, s, L, rho, th, dm, alpNSI, PrNSI 
            );
            file_results << std::setprecision(2) << ene << ","<< std::scientific
                << std::setprecision(6)
                << PrSTD[1][0] << "," 
                << PrINV[1][0] << "," 
                << PrVEP[1][0] << ","
                << PrNSI[1][0] << "\n" << std::fixed;
            ene+=0.01;
        }
    }
    file_results.close();
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop_time - start_time;
    std::cout << "Total time: " << duration.count() << " ms\n" ;
    delete[] PrNSI;
    delete[] PrSTD;
    delete[] PrVEP;
    delete[] PrINV;

    return 0;
}