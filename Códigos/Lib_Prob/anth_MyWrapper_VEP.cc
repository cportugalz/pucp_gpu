#include "anth_MyClass_VEP.h"
#include "anth_MyWrapper_VEP.h"

extern "C" {
        MiClaseVEP* Anth_VEP(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double gamma[3], double P[3][3])
	{
                return new MiClaseVEP(Energ,sigN,L,rhomat,th,dCP,dm,gamma,P);
        }

        void delete_Anth_VEP(MiClaseVEP* v) {
                delete v;
        }
}
