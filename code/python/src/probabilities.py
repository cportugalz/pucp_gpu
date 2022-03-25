import numpy as np
import scipy as sp
import math
from utils import ProbConst, PhiIntegration

def InvisibleDecay(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P):

    Pot = np.array((3,3))
    Hff = np.array((3,3))
    S = np.array((3,3))
    V = np.array((3,3))

    energy = _energy * 1.e9
    rho = _sigN * _rho

    DM = np.zeros((3,3),dtype=np.complex)

    DM[0,0] = np.complex(0, -0.5 * _alpha[0] / energy)
    DM[1,1] = np.complex(0.5 * _dm[0] / energy, -0.5 * _alpha[1] / energy)
    DM[2,2] = np.complex(0.5 * _dm[1] / energy, -0.5 * _alpha[2] / energy)

    Pot << np.complex(rho * 7.63247 * 0.5 * 1e-14, 0), DM[0][1], DM[0][2], DM[1][0], DM[0][0], DM[1][2], DM[2][0], DM[2][1], DM[0][0]

    UPMNS = np.array((3,3))
    Hd = np.array((3,3))

    UPMNS << _U[1][0], _U[0][1], _U[0][2], _U[1][0], _U[1][1], _U[1][2], _U[2][0], _U[2][1], _U[2][2]
    Hd << DM[0][0], DM[0][1], DM[0][2], DM[1][0], DM[1][1], DM[1][2], DM[2][0], DM[2][1], DM[2][2]

    Hff = UPMNS * Hd * UPMNS.conj().T + Pot

    tmp = sp.linalg.eig(Hff)

    V = tmp[1]

    S << np.exp(-ProbConst.I * tmp[0][0] * _L * 1e9 / ProbConst.GevkmToevsq), DM[0][0], DM[0][0], DM[0][0], np.exp(-ProbConst.I * tmp[0][1] * _L * 1e9 / ProbConst.GevkmToevsq), DM[0][0], DM[0][0], DM[0][0], np.exp(-ProbConst.I * tmp[0][2] * _L * 1e9 / ProbConst.GevkmToevsq)
    S = (V) * S * (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,i][j] * _P[:,i][j])

def StandardOscilation(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P):
    
    InvisibleDecay(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P)

def NonStandardInteraction(_U, _energy, _sigN, _L, _rho, _dm, _parmNSI, _P):

    Pot = np.array((3,3))
    MNSI = np.array((3,3))
    Hff = np.array((3,3))
    S = np.array((3,3))
    V = np.array((3,3))

    energy = _energy * 1.e9
    rho = _sigN * _rho

    DM = np.zeros((3,3), dtype=np.complex)
    NSI = np.array((3,3), dtype=np.complex)

    DM[1,1] = np.complex(0,5 * _dm[0] / energy, 0)
    DM[2,2] = np.complex(0,5 * _dm[1] / energy, 1)

    NSI[0,0] = _parmNSI[0]
    NSI[1,1] = _parmNSI[3] * np.exp(ProbConst.I * _parmNSI[4])
    NSI[0,2] = _parmNSI[5] * np.exp(ProbConst.I * _parmNSI[6])
    NSI[1,0] = _parmNSI[3] * np.exp(-ProbConst.I * _parmNSI[4])
    NSI[1,1] = _parmNSI[1]
    NSI[1,2] = _parmNSI[7] * np.exp(ProbConst.I * _parmNSI[8])
    NSI[2,0] = _parmNSI[5] * np.exp(-ProbConst.I * _parmNSI[6])
    NSI[2,1] = _parmNSI[7] * np.exp(-ProbConst.I * _parmNSI[8])
    NSI[2,2] = _parmNSI[2]

    Pot << np.complex(rho * 7.63247 * 0.5 * 1e-14, 0), DM[0][1], DM[0][2], DM[1][0], DM[0][0], DM[1][2], DM[2][0], DM[2][1], DM[0][0]
    MNSI << NSI[0,0], NSI[0,1], NSI[0,2], NSI[1,0], NSI[1,1], NSI[1,2], NSI[2,0], NSI[2,1], NSI[2,2]

    UPMNS = np.array((3,3))
    UPMNS << _U[0,0], _U[0,1], _U[0,2], _U[1,0], _U[1,1], _U[1,2], _U[2,0], _U[2,1], _U[2,2]

    Hd = np.array((3,3))
    Hd << DM[0][0], DM[0][1], DM[0][2], DM[1][0], DM[1][1], DM[1][2], DM[2][0], DM[2][1], DM[2][2]

    Hff = UPMNS * Hd * UPMNS.conj().T + rho * 7.63247 * 0.5 * 1e-14 * MNSI + Pot

    tmp = sp.linalg.eig(Hff)

    V = tmp[1]

    S << np.exp(-ProbConst.I * tmp[0][0] * _L * 1e9 / ProbConst.GevkmToevsq), DM[0][0], DM[0][0], DM[0][0], np.exp(-ProbConst.I * tmp[0][1] * _L * 1e9 / ProbConst.GevkmToevsq), DM[0][0], DM[0][0], DM[0][0], np.exp(-ProbConst.I * tmp[0][2] * _L * 1e9 / ProbConst.GevkmToevsq)
    S = (V) * S * (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,i][j] * _P[:,i][j])

def ViolationPrincipleDecay(_U, _energy, _sigN, _L, _rho, _dm, _gamma, _P):

    Pot = np.array((3,3))
    Hff = np.array((3,3))
    S = np.array((3,3))
    V = np.array((3,3))

    energy = _energy * 1.e9
    rho = _sigN * _rho

    DM = np.zeros((3,3), dtype=np.complex)

    DM[0,0] = np.complex(2 * energy * _gamma[0], 0)
    DM[1,1] = np.complex(0,5 * _dm[0] / energy + 2 * energy * _gamma[1], 0)
    DM[2,2] = np.complex(0,5 * _dm[1] / energy + 2 * energy * _gamma[2], 0)

    Pot << np.complex(rho * 7.63247 * 0.5 * 1e-14, 0), DM[0][1], DM[0][2], DM[1][0], DM[0][0], DM[1][2], DM[2][0], DM[2][1], DM[0][0]
    
    UPMNS = np.array((3,3))

    UPMNS << _U[0,0], _U[0,1], _U[0,2], _U[1,0], _U[1,1], _U[1,2], _U[2,0], _U[2,1], _U[2,2]

    Hd = np.array((3,3))

    Hd << DM[0][0], DM[0][1], DM[0][2], DM[1][0], DM[1][1], DM[1][2], DM[2][0], DM[2][1], DM[2][2]

    Hff = UPMNS * Hd * UPMNS.conj().T + Pot

    tmp = sp.linalg.eig(Hff)

    V = tmp[1]

    S << np.exp(-ProbConst.I * tmp[0][0] * _L * 1e9 / ProbConst.GevkmToevsq), DM[0][0], DM[0][0], DM[0][0], np.exp(-ProbConst.I * tmp[0][1] * _L * 1e9 / ProbConst.GevkmToevsq), DM[0][0], DM[0][0], DM[0][0], np.exp(-ProbConst.I * tmp[0][2] * _L * 1e9 / ProbConst.GevkmToevsq)

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,i][j] * _P[:,i][j])

def ProbabilityVis(_energy, _L, _rho, _th, _dm, d, _alpha, _mlight, _tfi, _tsi, _tff, _tsf, _tpar, _thij, _tqcoup, _P):

    Pot = np.array((3,3))

    Enf = _energy * 1.e9
    mss = np.array((3))
    
    if (_dm[1] > 0):
        mss[0] = _mlight
        mss[1] = _dm[0] + _mlight
        mss[2] = _dm[1] + _mlight
    else:
        mss[0] = _mlight
        mss[1] = -_dm[1] + _mlight
        mss[2] = _dm[0] - _dm[1] + _mlight
    
    mm = min(20, (mss[_tpar] / mss[_thij]) * _energy)

    Pot << np.complex(_rho * 7.63247 * 0.5 * 1e-14, 0), ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, ProbConst.Z0
    
    _P = 1.e9 * np.polynomial.legendre.leggauss(PhiIntegration())