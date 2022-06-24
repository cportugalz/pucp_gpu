#include <SQuIDS/SQuIDS.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#ifndef __MYCLASSDCH_H
#define __MYCLASSDCH_H


class MiClaseDCH: public squids::SQuIDS {
 private:

	double Decoh[9][9];
    std::unique_ptr<squids::SU_vector[]> b0_proj;
    squids::SU_vector DM2;
    squids::SU_vector PotA;
    std::vector<squids::SU_vector> Pot_evol;

	squids::SU_vector estadofnl;
	int signoN;

	void PreDerive(double t){
      for(unsigned int ei = 0; ei < nx; ei++){
        squids::SU_vector h0 = H0( Get_x(ei) , 0);  
        Pot_evol[ei] = PotA.Evolve(h0,t-Get_t_initial());
      }
	}


    squids::SU_vector InteractionsRho(unsigned int ei, unsigned int index_rho, double t) const{
	  squids::SU_vector GammaRhoSU = squids::SU_vector::Generator(3,0);
	  GammaRhoSU = estate[ei].rho[index_rho];
      for ( int i = 0 ; i < 9; i++){
        GammaRhoSU[i] = Decoh[i][i] * GammaRhoSU[i] ; 
	  }
      return  -GammaRhoSU ;} 

    squids::SU_vector H0(double x, unsigned int irho) const{
    return DM2*(0.5/x); 
	}

    squids::SU_vector HI(unsigned int ix, unsigned int irho, double t) const{
    return   Pot_evol[ix]; 
	}

 public:
  MiClaseDCH(double Energ, /*int fini,*/ int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double gmr[9], double P[3][3])
	{	
    	squids::Const units;
	unsigned int Nbins=1;
	ini(Nbins/*nodes*/,3/*SU(3)*/,1/*density matrices*/,0/*scalars*/,0/*tiempo inicial*/);
	Set_xrange(Energ*1.e9, Energ*1.e9, "lin"); // log
	signoN = sigN;
    	params.SetMixingAngle(0,1,th[0]);
    	params.SetMixingAngle(0,2,th[1]);
    	params.SetMixingAngle(1,2,th[2]);
    	params.SetPhase(0,2,sigN*dCP);
    	params.SetEnergyDifference(1,dm[0]);
    	params.SetEnergyDifference(2,dm[1]);

    	Set_OtherRhoTerms(true);
		Set_CoherentRhoTerms(true);

  	b0_proj.reset(new squids::SU_vector[3]);
  	for(int i = 0; i < 3; i++){
    		b0_proj[i]=squids::SU_vector::Projector(3,i);
  	}

  	DM2=squids::SU_vector(3);
  	for(int i = 1; i < 3; i++)
    	{DM2 += (b0_proj[i])*params.GetEnergyDifference(i);}

	double potenc_matt = signoN*rhomat*7.63247*0.5*1.e-14; // 2.956740
	PotA = squids::SU_vector(3);
	PotA = b0_proj[0]*potenc_matt;
	PotA.RotateToB1(params);

	// Errores GSL 
  	Set_rel_error(1e-7);
  	Set_abs_error(1e-7);
  	Set_h(1e-10);
  	Set_GSL_step(gsl_odeiv2_step_rk8pd);
  	
  	// Constante para conversión de unidades. Otra opción: units.km = 1.e9/GevkmToevsq
  	double GevkmToevsq = 0.197327; 
  	
  	// Definimos la matriz de disipación 9x9 diagonal
  	for ( int k = 0 ; k < 9; k++){
       	Decoh[k][k]=gmr[k]*units.GeV;
    	}
	
	// Calculamos la Matriz de Probabilidad
	for (int i = 0; i < 3 ; i++) {
	// Tiempo inicial del sistema de ecuaciones
	ini(Nbins/*nodes*/,3/*SU(3)*/,1/*density matrices*/,0/*scalars*/,0/*tiempo inicial*/);
	
	squids::SU_vector estadoini;
	estadoini = squids::SU_vector::Projector(3, i );
    	estadoini.RotateToB1(params);

    	for ( int ei = 0; ei < Nbins; ei++ ){
		state[ei].rho[0] = estadoini; }

	///// Jugada para PreDerive
    	Pot_evol.resize(nx);
    	for(int ei = 0; ei < nx; ei++){
    	  Pot_evol[ei] = squids::SU_vector(3); }

	// Evolucionamos el sistema de ecuaciones el tiempo L
   	Evolve(L*1.e9/GevkmToevsq);
   	
   	// Calculo de Probabilidad
	for (int j = 0; j < 3 ; j++) {
   	estadofnl = squids::SU_vector::Projector(3, j );
   	estadofnl.RotateToB1(params);
   	
   	P[i][j] = GetExpectationValue(estadofnl,0, 0 /* Bin 0 de energia */);
   	
	} }

	}


};

#endif
