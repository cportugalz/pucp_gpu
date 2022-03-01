
#ifndef __MYWRAPPERVIS_H
#define __MYWRAPPERVIS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiClaseVIS MiClaseVIS;

MiClaseVIS* Anth_VIS(double E, double Long, double rho, double th[3], double dCP, double dm[2], double mlight, double alpha[3], int tfi, int tsi, int tff, int tsf, int tpar, int thij, int tqcoup, double P[1]);

void delete_Anth_VIS(MiClaseVIS* v) ;

#ifdef __cplusplus
}
#endif
#endif
