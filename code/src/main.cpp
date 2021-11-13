#include "Probabilities.h"
#include <iostream>
#include <fstream>
#include <chrono>


int main(){
    // Parmámetros para Oscilación Estándar
    double d = -1.57;
    double dm21 = 7.4e-5;
    double dm31 = 2.5e-3;
    double L = 1300; double rho = 2.956740;
    int s = 1; int a = 1 , b = 0 ;

    // Parámetros para Inv. Decay
    double alp2 = 0 ; double alp3 = 5.e-5;

    // Parámetros para VEP
    double gam2 = 0 ; double gam3 = 2.e-24;

    // Parámetros para NSI
    double ee = 0 ; double mm = 0; double tt = 0;
    double em = 0.05 ; double emf = -1.55;
    double et = 0 ; double etf = 0;
    double mt = 0 ; double mtf = 0;

    // Calculamos las tablas de probabilidad
    // std::cout << "Making probabilities data\n";
    // FILE *temptext; double ene;
    // char tmpS[5]; sprintf(tmpS, "%i", s);
    // char nombre[200] = "Tablas/Probabilidad_s_"; strcat( nombre, tmpS  ); strcat( nombre, ".dat" );
    // temptext=fopen(nombre,"w");

    // fprintf(temptext, "Energia	P_OscStd	P_InvDcy	P_VEP	P_NSI	\n");
    Probabilities simulation;
    double th[] = { 0.59, 0.15, 0.84 };
    double dm[] = { 7.4e-5, 2.5e-3 }; 
    double alpSTD[3] = { 0 };
    double alpINV[3] = { 0, 0, 5.e-5 };
    double alpVEP[3] = { 0, 0, 2.e-24 };
    double alpNSI[] = { ee, mm, tt, em, emf, et, etf, mt, mtf };
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
        for(double ene=0.01 ; ene<=10; ene+=0.01) {
            simulation.StandardOscilation(
                ene, s, L, rho, th, d, dm, alpSTD, PrSTD
            );
            simulation.InvisibleDecay(
                ene, s, L, rho, th, d, dm, alpINV, PrINV 
            );
            simulation.ViolationPrincipleDecay(
                ene, s, L, rho, th, d, dm, alpVEP, PrVEP 
            );        
            simulation.NonStandardInteraction(
                ene, s, L, rho, th, d, dm, alpNSI, PrNSI 
            );
            file_results << std::scientific 
                << PrSTD[1][0] << "\t" 
                << PrINV[1][0] << "\t" 
                << PrVEP[1][0] << "\t"
                << PrNSI[1][0] << "\n" ;
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