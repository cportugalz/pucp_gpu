#include "anth_MyClass_INV.h"
#include "anth_MyWrapper_INV.h"

extern "C" {
        MiClaseINV* Anth_INV(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double alpha[3], double P[3][3])
	{
                return new MiClaseINV(Energ,sigN,L,rhomat,th,dCP,dm,alpha, P);
        }

        void delete_Anth_INV(MiClaseINV* v) {
                delete v;
        }
}
