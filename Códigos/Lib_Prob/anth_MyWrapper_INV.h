
#ifndef __MYWRAPPERINV_H
#define __MYWRAPPERINV_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiClaseINV MiClaseINV;

MiClaseINV* Anth_INV(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double alpha[3], double P[3][3]);

void delete_Anth_INV(MiClaseINV* v) ;

#ifdef __cplusplus
}
#endif
#endif
