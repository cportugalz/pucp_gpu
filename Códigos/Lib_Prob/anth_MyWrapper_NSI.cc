#include "anth_MyClass_NSI.h"
#include "anth_MyWrapper_NSI.h"

extern "C" {
        MiClaseNSI* Anth_NSI(double Energ, int sigN, double L, double rhomat, double th[3], double dCP, double dm[2], double parmNSI[9], double P[3][3])
	{
                return new MiClaseNSI(Energ,sigN,L,rhomat,th,dCP,dm,parmNSI, P);
        }

        void delete_Anth_NSI(MiClaseNSI* v) {
                delete v;
        }
}
