
#ifndef __MYWRAPPERNSI_H
#define __MYWRAPPERNSI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiClaseNSI MiClaseNSI;

MiClaseNSI* Anth_NSI(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double parmNSI[9], double P[3][3]);

void delete_Anth_NSI(MiClaseNSI* v) ;

#ifdef __cplusplus
}
#endif
#endif
