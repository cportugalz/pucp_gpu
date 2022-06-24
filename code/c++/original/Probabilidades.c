#include <stdio.h>
#include <string.h> // Para las funciones de exportacion de datos

//Librerias de Probabilidad
#include "Lib_Prob/anth_MyWrapper_INV.h"
#include "Lib_Prob/anth_MyWrapper_VEP.h"
#include "Lib_Prob/anth_MyWrapper_NSI.h"
#include "Lib_Prob/anth_MyWrapper_VIS.h"
#include "Lib_Prob/anth_MyWrapper_DCH.h"


// Matrices temporales
double PrSTD[3][3];
double PrINV[3][3];
double PrVEP[3][3];
double PrNSI[3][3];
double PrDCH[3][3];


// Funciones que calculan y escriben las Matrices de Probabilidad
void MatrizProbSTD (double E, int s, double L, double rho, double th12, double th13, double th23, double delta, double dm21, double dm31, double P[3][3])
{ double th[3] = {th12, th13, th23}; double dm[2]= {dm21, dm31}; double alp[3]= {0,0,0};
  struct MiClaseINV* cinv = Anth_INV(E, s, L, rho, th, delta, dm, alp, P);
  delete_Anth_INV(cinv);
}

void MatrizProbINV (double E, int s, double L, double rho, double th12, double th13, double th23, double delta, double dm21, double dm31, double a2, double a3, double P[3][3])
{ double th[3] = {th12, th13, th23}; double dm[2]= {dm21, dm31}; double alp[3]= {0,a2,a3};
  struct MiClaseINV* cinv = Anth_INV(E, s, L, rho, th, delta, dm, alp, P);
  delete_Anth_INV(cinv);
}

void MatrizProbVEP (double E, int s, double L, double rho, double th12, double th13, double th23, double delta, double dm21, double dm31, double g2, double g3, double P[3][3])
{ double th[3] = {th12, th13, th23}; double dm[2]= {dm21, dm31}; double gam[3]= {0,g2,g3};
  struct MiClaseVEP* cvep = Anth_VEP(E, s, L, rho, th, delta, dm, gam, P);
  delete_Anth_VEP(cvep);
}

void MatrizProbNSI (double E, int s, double L, double rho, double th12, double th13, double th23, double delta, double dm21, double dm31, double ee, double uu, double tt, double em, double emf, double et, double etf, double mt, double mtf, double P[3][3])
{ double th[3] = {th12, th13, th23}; double dm[2]= {dm21, dm31}; double nsi[9]= {ee,uu,tt,em,emf,et,etf,mt,mtf};
  struct MiClaseNSI* cnsi = Anth_NSI(E, s, L, rho, th, delta, dm, nsi, P);
  delete_Anth_NSI(cnsi);
}

void ProbVIS (double E, double L, double rho, double th12, double th13, double th23, double delta, double dm21, double dm31, double mlight, double a2, double a3, int tfi, int tsi, int tff, int tsf, int tpar, int thij, int tqcoup, double P[1])
{ double th[3] = {th12, th13, th23}; double dm[2]= {dm21, dm31}; double alp[3]= {0,a2,a3};
  struct MiClaseVIS* cvis = Anth_VIS(E, L, rho, th, delta, dm, mlight, alp, tfi, tsi, tff, tsf, tpar, thij, tqcoup, P);
  delete_Anth_VIS(cvis);
}

void MatrizProbDCH (double E, int s, double L, double rho, double th12, double th13, double th23, double delta, double dm21, double dm31, double gamma_d, double P[3][3])
{ /// Decoherencia SQuIDS
  double Grm1 = gamma_d;//*1.e-24;
  double Grm2 = Grm1*5./3.;//;Grm1/4;//Grm1*5./3.
  //double GmrC[9] = {0,Grm1,Grm2,Grm1,0,Grm2,Grm2,Grm2,0}; // Orden para Squids Modelo A
  double GmrC[9] = {0,Grm1,Grm1,Grm2,Grm1,Grm1,Grm2,Grm2,Grm1}; // Orden para Squids Modelo B
  double thmix[3] = {th12,th13,th23}; double deltaCP = delta; double dmq[2] = {dm21, dm31};
   
   struct MiClaseDCH* cdch = Anth_DCH(E, s, L, rho, thmix, deltaCP, dmq, GmrC, P);
   delete_Anth_DCH(cdch);
}

/***************************************************************/
/******************   Programa Principal  **********************/
/***************************************************************/

int main ( )
{
 // Parmámetros para Oscilación Estándar
 double th12 = 0.59; double d = -1.57;
 double th13 = 0.15; double dm21 = 7.4e-5;
 double th23 = 0.84; double dm31 = 2.5e-3;
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
 
 // Parámetros para Vis. Decay
 int fi_1 = 1, si_1 = 1; int fi_2 = 0, si_2 = 1;
 int ff_1 = 0, sf_1 = 1; int ff_2 = 1, sf_2 = -1;
 int par_1 = 2, hij_1 = 0; int par_2 = 1, hij_2 = 2;
 int qcoup = 1; double mlight = 0.05*0.05; // 1e-20 0.05*0.05
 double PrVis_1[1], PrVis_2[1];
 
 // Parámetros para Decoherencia
 double Gamma = 2.e-24 ;
 
 
 // Calculamos las tablas de probabilidad
 printf("\n Generando tabla de probabilidad... \n");
   FILE *temptext; double ene;
   char tmpS[5]; sprintf(tmpS, "%i", s);
//   char nombre[200] = "Tablas/Probabilidad_s_"; strcat( nombre, tmpS  ); strcat( nombre, ".dat" );
	char nombre[200] = "Tablas/Probabilidad_todo_"; strcat( nombre, tmpS  ); strcat( nombre, ".dat" );
   temptext=fopen(nombre,"w");

   fprintf(temptext, "Energia	P_OscStd	P_InvDcy	P_VEP	P_NSI	P_VIS_1	P_VIS_2	P_DCH	\n");
   for(ene=0.01 ; ene<=10 ; ene=ene+0.01)
   {
 	MatrizProbSTD (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, PrSTD );
 	MatrizProbINV (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, alp2, alp3, PrINV );
 	MatrizProbVEP (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, gam2, gam3, PrVEP );
 	MatrizProbNSI (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, ee, mm, tt, em, emf, et, etf, mt, mtf, PrNSI );
 	ProbVIS (ene, L, rho, th12, th13, th23, d, dm21, dm31, mlight, alp2, alp3, fi_1, si_1, ff_1, sf_1, par_1, hij_1, qcoup, PrVis_1);
 	ProbVIS (ene, L, rho, th12, th13, th23, d, dm21, -dm31, mlight, alp2, alp3, fi_2, si_2, ff_2, sf_2, par_2, hij_2, qcoup, PrVis_2);
 	MatrizProbDCH (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, Gamma, PrDCH );

	fprintf(temptext, "%g	%.8g	%.8g	%.8g	%.8g	%.8g	%.8g	%.8g	\n", ene, PrSTD[a][b], PrINV[a][b], PrVEP[a][b], PrNSI[a][b], PrVis_1[0], PrVis_2[0], PrDCH[a][b] );
	
   } 
   fclose(temptext);
   printf("\n ... Finalizó tabla de probabilidad en C/C++ con s = %i . \n", s);

 return 0;
}
