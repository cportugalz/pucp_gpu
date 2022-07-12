
#ifndef __MYWRAPPERDCH_H
#define __MYWRAPPERDCH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiClaseDCH MiClaseDCH;

MiClaseDCH* Anth_DCH(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double gmr[9], double P[3][3]);

void delete_Anth_DCH(MiClaseDCH* v) ;

#ifdef __cplusplus
}
#endif
#endif
