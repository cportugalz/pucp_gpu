
#ifndef __MYWRAPPERVEP_H
#define __MYWRAPPERVEP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiClaseVEP MiClaseVEP;

MiClaseVEP* Anth_VEP(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double gamma[3], double P[3][3]);

void delete_Anth_VEP(MiClaseVEP* v) ;

#ifdef __cplusplus
}
#endif
#endif
