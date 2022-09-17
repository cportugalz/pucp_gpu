#include "probabilities.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip> // Compatibility  with linux gcc
#include <complex>


int main(int argc, char* argv[]){
	if(argc < 2){
		std::cout << "Usage: make single n=<your number>" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::string sim_arg(argv[1]);
	int num_simulations = std::stod(sim_arg);
	// Standard Oscilation
	double d = -1.57;
	double L = 1300; double rho = 2.956740;
	int s = 1; int a = 1 , b = 0 ;
	double th[] = { 0.59, 0.15, 0.84 };
	
	double dm[] = { 7.4e-5, 2.5e-3 }; 
	double alpSTD[3] = { 0 };
	// Invisible decay
	double alpINV[3] = { 0, 0, 5.e-3 }; // (0, alpha2, alpha3))
	// VEP
	//double alpVEP[3] = { 0, 0, 2.e-24}; //(Gamma1, Gamma2, Gamma2))
	double alpVEP[3] = { 0, 4.e-24, 0}; //(Gamma1, Gamma2, Gamma2))
	// Non Standard Interaction
	// ee, mm,tt pertenecen a Reales
	// em, emf -> modulo y fase
	// et, etf -> modulo y fase 
	// mt, mtf -> "           "
	double ee = 0 ; double mm = 0; double tt = 0;
	double em = 0.05 ; double emf = -1.55;
	double et = 0 ; double etf = 0;
	double mt = 0 ; double mtf = 0;
	double alpNSI[] = { ee, mm, tt, em, emf, et, etf, mt, mtf };
	double delta = s * d;
	// Visible decay?
	int fi_1 = 1, si_1 = 1; int fi_2 = 0, si_2 = 1;
	int ff_1 = 0, sf_1 = 1; int ff_2 = 1, sf_2 = -1;
	int par = 2, hij = 0;
	int qcoup = 1; double mlight = 0.05*0.05; // 1e-20 0.05*0.05
	double PrVis_1, PrVis_2;

	double** PrSTD = new double*[3];
	double** PrINV = new double*[3];
	double** PrVEP = new double*[3];
	double** PrNSI = new double*[3];
	double** PrDCH = new double*[3];
	for ( int i=0; i<3; ++i){
		PrSTD[i] = new double[3];
		PrINV[i] = new double[3];
		PrVEP[i] = new double[3];
		PrNSI[i] = new double[3];
		PrDCH[i] = new double[3];
	}
	// for standard oscilation, invisible decay, vpd and nsi
	std::complex<double>** U1 = make_umns(s, th, d);
	std::ofstream file_results("output/output.txt");
	auto start_time = std::chrono::high_resolution_clock::now();
	if(file_results.is_open()){        
		for(int iter_energy=1; iter_energy<=num_simulations; ) {
			double energy = iter_energy/100.0;
			// StandardOscilation(
			// 	U1, energy, s, L, rho, dm, alpSTD, PrSTD
			// );
			// InvisibleDecay(
			// 	U1, energy, s, L, rho, dm, alpINV, PrINV 
			// );
			// ViolationEquivalencePrinciple(
			// 	U1, energy, s, L, rho, dm, alpVEP, PrVEP 
			// );        
			NonStandardInteraction(
				U1, energy, s, L, rho, dm, alpNSI, PrNSI 
			);
			// VisibleDecay(
			// 	energy, L, rho, th, dm, d, alpINV, mlight, 
			// 	fi_1, si_1, ff_1, sf_1, par, hij, qcoup, &PrVis_1
			// );
			// VisibleDecay(
			// 	energy, L, rho, th, dm, d, alpINV, mlight, 
			// 	fi_2, si_2, ff_2, sf_2, par, hij, qcoup, &PrVis_2
			// );
				
			file_results << std::setprecision(2) << energy << ","<< std::fixed
				<< std::setprecision(8)
				<< PrSTD[1][0] << "," 
				<< PrINV[1][0] << "," 
				<< PrVEP[1][0] << ","
				<< PrNSI[1][0] << ","
				<< PrVis_1 << "," 
				<< PrVis_2 << "\n" << std::fixed;
			iter_energy += 1;
		}
	}
	file_results.close();
	auto stop_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = stop_time - start_time;
	std::cout << "Total time: " << duration.count()/1000 << " s\n" ;
	delete[] PrNSI;
	delete[] PrSTD;
	delete[] PrVEP;
	delete[] PrINV;

	return 0;
}