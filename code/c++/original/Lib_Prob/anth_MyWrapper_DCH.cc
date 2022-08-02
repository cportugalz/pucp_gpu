#include "anth_MyClass_DCH.h"
#include "anth_MyWrapper_DCH.h"

extern "C" {
        MiClaseDCH* Anth_DCH(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double gmr[9], double P[3][3])
	{
                return new MiClaseDCH(Energ,sigN,L,rhomat,th,dCP,dm,gmr, P);
        }

        void delete_Anth_DCH(MiClaseDCH* v) {
                delete v;
        }
}
