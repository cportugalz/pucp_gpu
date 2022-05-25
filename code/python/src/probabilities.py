import numpy as np
import scipy.optimize
import math
from utils import ProbConst, PhiIntegration, dotComplexMatrix

def InvisibleDecay(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P):

    Pot = np.ndarray((3,3),dtype=np.complex)
    Hff = np.ndarray((3,3),dtype=np.complex)
    S = np.ndarray((3,3),dtype=np.complex)
    V = np.ndarray((3,3),dtype=np.complex)

    energy = _energy * 1.e9
    rho = _sigN * _rho

    DM = np.zeros((3,3),dtype=np.complex)

    DM[0,0] = np.complex(0, -0.5 * _alpha[0] / energy)
    DM[1,1] = np.complex(0.5 * _dm[0] / energy, -0.5 * _alpha[1] / energy)
    DM[2,2] = np.complex(0.5 * _dm[1] / energy, -0.5 * _alpha[2] / energy)

    Pot[0,0] = np.complex(rho * 7.63247 * 0.5 * 1e-14, 0)
    Pot[0,1] = DM[0][1]
    Pot[0,2] = DM[0][2]
    Pot[1,0] = DM[1][0]
    Pot[1,1] = DM[0][0]
    Pot[1,2] = DM[1][2]
    Pot[2,0] = DM[2][0]
    Pot[2,1] = DM[2][1]
    Pot[2,2] = DM[0][0]

    UPMNS = np.ndarray((3,3),dtype=np.complex)
    HD = np.ndarray((3,3),dtype=np.complex)

    UPMNS[0,0] = _U[0][0]
    UPMNS[0,1] = _U[0][1]
    UPMNS[0,2] = _U[0][2]
    UPMNS[1,0] = _U[1][0]
    UPMNS[1,1] = _U[1][1]
    UPMNS[1,2] = _U[1][2]
    UPMNS[2,0] = _U[2][0]
    UPMNS[2,1] = _U[2][1]
    UPMNS[2,2] = _U[2][2]

    HD[0,0] = DM[0][0]
    HD[0,1] = DM[0][1]
    HD[0,2] = DM[0][2]
    HD[1,0] = DM[1][0]
    HD[1,1] = DM[1][1]
    HD[1,2] = DM[1][2]
    HD[2,0] = DM[2][0]
    HD[2,1] = DM[2][1]
    HD[2,2] = DM[2][2]

    N = UPMNS @ HD
    A = dotComplexMatrix(N, UPMNS.transpose())

    Hff = A + Pot

    tmp = np.linalg.eigh(Hff)

    V = tmp[1]

    S[0,0] = np.exp(-ProbConst.I * tmp[0][0] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[0,1] = DM[0][0]
    S[0,2] = DM[0][0]
    S[1,0] = DM[0][0]
    S[1,1] = np.exp(-ProbConst.I * tmp[0][1] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[1,2] = DM[0][0]
    S[2,0] = DM[0][0]
    S[2,1] = DM[0][0]
    S[2,2] = np.exp(-ProbConst.I * tmp[0][2] * _L * 1e9 / ProbConst.GevkmToevsq)
    S = (V) @ S @ (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,j][i] * S[:,j][i])
    

def StandardOscilation(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P):
    
    InvisibleDecay(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P)

def NonStandardInteraction(_U, _energy, _sigN, _L, _rho, _dm, _parmNSI, _P):

    Pot = np.ndarray((3,3))
    MNSI = np.ndarray((3,3))
    Hff = np.ndarray((3,3))
    S = np.ndarray((3,3))
    V = np.ndarray((3,3))

    energy = _energy * 1.e9
    rho = _sigN * _rho

    DM = np.zeros((3,3), dtype=np.complex)
    NSI = np.ndarray((3,3), dtype=np.complex)

    DM[1,1] = np.complex(0.5 * _dm[0] / energy, 0)
    DM[2,2] = np.complex(0.5 * _dm[1] / energy, 1)

    NSI[0,0] = _parmNSI[0]
    NSI[1,1] = _parmNSI[3] * np.exp(ProbConst.I * _parmNSI[4])
    NSI[0,2] = _parmNSI[5] * np.exp(ProbConst.I * _parmNSI[6])
    NSI[1,0] = _parmNSI[3] * np.exp(-ProbConst.I * _parmNSI[4])
    NSI[1,1] = _parmNSI[1]
    NSI[1,2] = _parmNSI[7] * np.exp(ProbConst.I * _parmNSI[8])
    NSI[2,0] = _parmNSI[5] * np.exp(-ProbConst.I * _parmNSI[6])
    NSI[2,1] = _parmNSI[7] * np.exp(-ProbConst.I * _parmNSI[8])
    NSI[2,2] = _parmNSI[2]

    Pot[0,0] = np.complex(rho * 7.63247 * 0.5 * 1e-14, 0)
    Pot[0,1] = DM[0][1]
    Pot[0,2] = DM[0][2]
    Pot[1,0] = DM[1][0]
    Pot[1,1] = DM[0][0]
    Pot[1,2] = DM[1][2]
    Pot[2,0] = DM[2][0]
    Pot[2,1] = DM[2][1]
    Pot[2,2] = DM[0][0]
    
    MNSI[0,0] = NSI[0,0]
    MNSI[0,1] = NSI[0,1]
    MNSI[0,2] = NSI[0,2]
    MNSI[1,0] = NSI[1,0]
    MNSI[1,1] = NSI[1,1]
    MNSI[1,2] = NSI[1,2]
    MNSI[2,0] = NSI[2,0]
    MNSI[2,1] = NSI[2,1]
    MNSI[2,2] = NSI[2,2]

    UPMNS = np.ndarray((3,3), dtype=np.complex)
    UPMNS[0,0] = _U[0,0]
    UPMNS[0,1] = _U[0,1]
    UPMNS[0,2] = _U[0,2]
    UPMNS[1,0] = _U[1,0]
    UPMNS[1,1] = _U[1,1]
    UPMNS[1,2] = _U[1,2]
    UPMNS[2,0] = _U[2,0]
    UPMNS[2,1] = _U[2,1]
    UPMNS[2,2] = _U[2,2]

    HD = np.ndarray((3,3), dtype=np.complex)
    HD[0,0] = DM[0][0]
    HD[0,1] = DM[0][1]
    HD[0,2] = DM[0][2]
    HD[1,0] = DM[1][0]
    HD[1,1] = DM[1][1]
    HD[1,2] = DM[1][2]
    HD[2,0] = DM[2][0]
    HD[2,1] = DM[2][1]
    HD[2,2] = DM[2][2]

    N = UPMNS @ HD
    A = dotComplexMatrix(N, UPMNS.transpose())

    Hff = A + rho * 7.63247 * 0.5 * 1e-14 @ MNSI + Pot

    tmp = sp.linalg.eigh(Hff)

    V = tmp[1]

    S[0,0] = np.exp(-ProbConst.I * tmp[0][0] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[0,1] = DM[0][0]
    S[0,2] = DM[0][0]
    S[1,0] = DM[0][0]
    S[1,1] = np.exp(-ProbConst.I * tmp[0][1] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[1,2] = DM[0][0]
    S[2,0] = DM[0][0]
    S[2,1] = DM[0][0]
    S[2,2] = np.exp(-ProbConst.I * tmp[0][2] * _L * 1e9 / ProbConst.GevkmToevsq)
    S = (V) @ S @ (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,j][i] * S[:,j][i])

def ViolationPrincipleDecay(_U, _energy, _sigN, _L, _rho, _dm, _gamma, _P):

    Pot = np.ndarray((3,3), dtype=np.complex)
    Hff = np.ndarray((3,3), dtype=np.complex)
    S = np.ndarray((3,3), dtype=np.complex)
    V = np.ndarray((3,3), dtype=np.complex)

    energy = _energy * 1.e9
    rho = _sigN * _rho

    DM = np.zeros((3,3), dtype=np.complex)

    DM[0,0] = np.complex(2 * energy * _gamma[0], 0)
    DM[1,1] = np.complex(0.5 * _dm[0] / energy + 2 * energy * _gamma[1], 0)
    DM[2,2] = np.complex(0.5 * _dm[1] / energy + 2 * energy * _gamma[2], 0)

    Pot[0,0] = np.complex(rho * 7.63247 * 0.5 * 1e-14, 0)
    Pot[0,1] = DM[0][1]
    Pot[0,2] = DM[0][2]
    Pot[1,0] = DM[1][0]
    Pot[1,1] = DM[0][0]
    Pot[1,2] = DM[1][2]
    Pot[2,0] = DM[2][0]
    Pot[2,1] = DM[2][1]
    Pot[2,2] = DM[0][0]
    
    UPMNS = np.ndarray((3,3), dtype=np.complex)

    UPMNS[0,0] = _U[0,0]
    UPMNS[0,1] = _U[0,1]
    UPMNS[0,2] = _U[0,2]
    UPMNS[1,0] = _U[1,0]
    UPMNS[1,1] = _U[1,1]
    UPMNS[1,2] = _U[1,2]
    UPMNS[2,0] = _U[2,0]
    UPMNS[2,1] = _U[2,1]
    UPMNS[2,2] = _U[2,2]

    HD = np.ndarray((3,3), dtype=np.complex)
    HD[0,0] = DM[0][0]
    HD[0,1] = DM[0][1]
    HD[0,2] = DM[0][2]
    HD[1,0] = DM[1][0]
    HD[1,1] = DM[1][1]
    HD[1,2] = DM[1][2]
    HD[2,0] = DM[2][0]
    HD[2,1] = DM[2][1]
    HD[2,2] = DM[2][2]

    N = UPMNS @ HD
    A = dotComplexMatrix(N, UPMNS.transpose())

    Hff = A + Pot

    tmp = scipy.linalg.eigh(Hff)

    V = tmp[1]

    S[0,0] = np.exp(-ProbConst.I * tmp[0][0] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[0,1] = DM[0][0]
    S[0,2] = DM[0][0]
    S[1,0] = DM[0][0]
    S[1,1] = np.exp(-ProbConst.I * tmp[0][1] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[1,2] = DM[0][0]
    S[2,0] = DM[0][0]
    S[2,1] = DM[0][0]
    S[2,2] = np.exp(-ProbConst.I * tmp[0][2] * _L * 1e9 / ProbConst.GevkmToevsq)

    S = (V) @ S @ (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,j][i] * S[:,j][i])

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