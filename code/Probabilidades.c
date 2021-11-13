#include <stdio.h>
#include <string.h> // Para las funciones de exportacion de datos
#include <chrono>
//Librerias de Probabilidad
#include "Lib_Prob/anth_MyWrapper_INV.h"
#include "Lib_Prob/anth_MyWrapper_VEP.h"
#include "Lib_Prob/anth_MyWrapper_NSI.h"

// Matrices temporales
double PrSTD[3][3];
double PrINV[3][3];
double PrVEP[3][3];
double PrNSI[3][3];

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
 
 // Calculamos las tablas de probabilidad
 printf("\n Generando tabla de probabilidad... \n");
   FILE *temptext; double ene;
   char tmpS[5]; sprintf(tmpS, "%i", s);
   char nombre[200] = "Tablas/Probabilidad_s_"; strcat( nombre, tmpS  ); strcat( nombre, ".dat" );
   temptext=fopen(nombre,"w");

   fprintf(temptext, "Energia	P_OscStd	P_InvDcy	P_VEP	P_NSI	\n");
  auto start_time = std::chrono::high_resolution_clock::now();

   for(ene=0.01 ; ene<=10 ; ene=ene+0.01)
   {
 	MatrizProbSTD (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, PrSTD );
 	MatrizProbINV (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, alp2, alp3, PrINV );
 	MatrizProbVEP (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, gam2, gam3, PrVEP );
 	MatrizProbNSI (ene, s, L, rho, th12, th13, th23, d, dm21, dm31, ee, mm, tt, em, emf, et, etf, mt, mtf, PrNSI );	

	fprintf(temptext, "%g	%.8f	%.8f	%.8f	%.8f	\n", ene, PrSTD[a][b], PrINV[a][b], PrVEP[a][b], PrNSI[a][b]);
   } 
   fclose(temptext);
   printf("\n ... Finalizó tabla de probabilidad en C/C++ con s = %d\n", s);
   auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop_time - start_time;
    std::cout << "Total time: " << duration.count() << " ms\n" ;
 return 0;
}
