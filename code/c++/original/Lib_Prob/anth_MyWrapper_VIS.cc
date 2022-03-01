#include "anth_MyClass_VIS.h"
#include "anth_MyWrapper_VIS.h"

extern "C" {
        MiClaseVIS* Anth_VIS(double E, double Long, double rho, double th[3], double dCP, double dm[2], double mlight, double alpha[3], int tfi, int tsi, int tff, int tsf, int tpar, int thij, int tqcoup, double P[1])
	{
                return new MiClaseVIS(E, Long, rho, th, dCP, dm, mlight, alpha, tfi, tsi, tff, tsf, tpar, thij, tqcoup, P);
        }

        void delete_Anth_VIS(MiClaseVIS* v) {
                delete v;
        }
}
