#include "Probabilities.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip> // Compatibility  with linux gcc
#include <thread>

void perform_simulation(std::string _filename, double _start_sim, double _end_sim){
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
    for ( int i=0; i<3; ++i){
        PrSTD[i] = new double[3];
        PrINV[i] = new double[3];
        PrVEP[i] = new double[3];
        PrNSI[i] = new double[3];
    }
    // for standard oscilation, invisible decay, vpd and nsi
	std::complex<double>** U1 = make_umns(s, th, d);
    std::ofstream file_results(_filename);
    if(file_results.is_open()){        
        for(double energy=_start_sim; int(energy*100)<=int(_end_sim*100); ) {            
            StandardOscilation(
                U1, energy, s, L, rho, dm, alpSTD, PrSTD
            );
            InvisibleDecay(
                U1, energy, s, L, rho, dm, alpINV, PrINV
            );
            ViolationPrincipleDecay(
                U1, energy, s, L, rho, dm, alpVEP, PrVEP
            );
            NonStandardInteraction(
                U1, energy, s, L, rho, dm, alpNSI, PrNSI
            );
            Probability_Vis(
				energy, L, rho, th, dm, d, alpINV, mlight, 
				fi_1, si_1, ff_1, sf_1, par, hij, qcoup, &PrVis_1
			);
			Probability_Vis(
				energy, L, rho, th, dm, d, alpINV, mlight, 
				fi_2, si_2, ff_2, sf_2, par, hij, qcoup, &PrVis_2
			);
            file_results << std::setprecision(2) << energy << ","<< std::fixed
                << std::setprecision(8)
                << PrSTD[1][0] << "," 
				<< PrINV[1][0] << "," 
				<< PrVEP[1][0] << ","
				<< PrNSI[1][0] << ","
				<< PrVis_1 << "," 
				<< PrVis_2 << "\n" << std::fixed;
            energy+=0.01;
        }
    }
    file_results.close();
    // std::cout<< "Thread " << std::this_thread::get_id() << " finished" << std::endl;
    delete[] PrNSI;
    delete[] PrSTD;
    delete[] PrVEP;
    delete[] PrINV;
}


int main(int argc, char* argv[]){

    if(argc < 3){
        std::cout << "Usage: make threaded n=<your number> t=<num. of threads>" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string argv_num_sim(argv[1]);
    std::string argv_num_threads(argv[2]);
    int num_simulations = std::stod(argv_num_sim);
    int num_threads = std::stod(argv_num_threads);
    if (num_threads % 2 != 0){
        std::cout<<"The number of threads must be even"<< std::endl;
        exit(EXIT_FAILURE);
    }

    // we must divide the data in num_sim/num_threads registres per each file
    int data_slice = num_simulations/num_threads;
    std::thread threads[num_threads];
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i=0; i<num_threads; i++){
        std::string filename = "output/output_" + std::to_string(i)+ ".txt";
        double start_index = (i*data_slice+1)/100.0;
        double stop_index = ((i+1)*data_slice)/100.0;
        threads[i] = std::thread(perform_simulation, filename, start_index, stop_index);
    }
    for (int i=0; i<num_threads; i++){
        threads[i].join();
    }

    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop_time - start_time;
    std::cout << "Total time: " << duration.count() << " ms\n" ;
    return 0;
}