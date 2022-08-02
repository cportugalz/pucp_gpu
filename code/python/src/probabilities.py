from time import process_time_ns
import numpy as np
from scipy.sparse.linalg import LinearOperator
from integrate.gaussLegendre import NintegrGLQ
from utils import ProbConst, PhiIntegration, matrixAdjoint

def InvisibleDecay(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P):

    Pot = np.ndarray((3,3),dtype=np.complex)
    Hff = np.ndarray((3,3),dtype=np.complex)
    S = np.ndarray((3,3),dtype=np.complex)
    V = np.ndarray((3,3),dtype=np.complex)

    energy = _energy * 1e9
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

    UPMNS = np.copy(_U)
    HD = np.copy(DM)
    T = np.copy(matrixAdjoint(UPMNS))

    N = UPMNS @ HD
    A = N @ T

    Hff = A + Pot

    eigvalues, eigvectors = np.linalg.eig(Hff)

    V = eigvectors

    S[0,0] = np.exp(-ProbConst.I * eigvalues[0] * _L * 1.e9 / ProbConst.GevkmToevsq)
    S[0,1] = DM[0][0]
    S[0,2] = DM[0][0]
    S[1,0] = DM[0][0]
    S[1,1] = np.exp(-ProbConst.I * eigvalues[1] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[1,2] = DM[0][0]
    S[2,0] = DM[0][0]
    S[2,1] = DM[0][0]
    S[2,2] = np.exp(-ProbConst.I * eigvalues[2] * _L * 1e9 / ProbConst.GevkmToevsq)

    S = (V) @ S @ (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,i][j] * S[:,i][j])
    

def StandardOscilation(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P):
    
    InvisibleDecay(_U, _energy, _sigN, _L, _rho, _dm, _alpha, _P)

def NonStandardInteraction(_U, _energy, _sigN, _L, _rho, _dm, _parmNSI, _P):
    # hermitian matrix

    Pot = np.ndarray((3,3), dtype=np.complex)
    MNSI = np.ndarray((3,3), dtype=np.complex)
    Hff = np.ndarray((3,3), dtype=np.complex)
    S = np.ndarray((3,3), dtype=np.complex)
    V = np.ndarray((3,3), dtype=np.complex)

    energy = _energy * 1e9
    rho = _sigN * _rho

    DM = np.zeros((3,3), dtype=np.complex)
    NSI = np.ndarray((3,3), dtype=np.complex)

    DM[1,1] = np.complex(0.5 * _dm[0] / energy, 0)
    DM[2,2] = np.complex(0.5 * _dm[1] / energy, 0)

    NSI[0,0] = _parmNSI[0]
    NSI[0,1] = _parmNSI[3] * np.exp(ProbConst.I * _parmNSI[4])
    NSI[0,2] = _parmNSI[5] * np.exp(ProbConst.I * _parmNSI[6])
    NSI[1,0] = _parmNSI[3] * np.exp(-ProbConst.I * _parmNSI[4])
    NSI[1,1] = _parmNSI[1]
    NSI[1,2] = _parmNSI[7] * np.exp(ProbConst.I * _parmNSI[8])
    NSI[2,0] = _parmNSI[5] * np.exp(-ProbConst.I * _parmNSI[6])
    NSI[2,1] = _parmNSI[7] * np.exp(-ProbConst.I * _parmNSI[8])
    NSI[2,2] = _parmNSI[2]

    Pot[0,0] = np.complex(rho * 7.63247 * 0.5 * 1.e-14, 0)
    Pot[0,1] = DM[0][1]
    Pot[0,2] = DM[0][2]
    Pot[1,0] = DM[1][0]
    Pot[1,1] = DM[0][0]
    Pot[1,2] = DM[1][2]
    Pot[2,0] = DM[2][0]
    Pot[2,1] = DM[2][1]
    Pot[2,2] = DM[0][0]

    MNSI = np.copy(NSI)

    UPMNS = np.copy(_U)

    HD = np.copy(DM)

    N = UPMNS @ HD
    T = np.copy(matrixAdjoint(UPMNS))
    A = N @ T

    Hff = A + (rho * 7.63247 * 0.5 * 1.e-14 * MNSI) + Pot

    eigvalues, eigvectors = np.linalg.eigh(Hff)

    V = eigvectors

    S[0,0] = np.exp(-ProbConst.I * eigvalues[0] * _L * 1.e9 / ProbConst.GevkmToevsq)
    S[0,1] = DM[0][0]
    S[0,2] = DM[0][0]
    S[1,0] = DM[0][0]
    S[1,1] = np.exp(-ProbConst.I * eigvalues[1] * _L * 1.e9 / ProbConst.GevkmToevsq)
    S[1,2] = DM[0][0]
    S[2,0] = DM[0][0]
    S[2,1] = DM[0][0]
    S[2,2] = np.exp(-ProbConst.I * eigvalues[2] * _L * 1.e9 / ProbConst.GevkmToevsq)
    S = (V) @ S @ (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,i][j] * S[:,i][j])

def ViolationPrincipleDecay(_U, _energy, _sigN, _L, _rho, _dm, _gamma, _P):
    # hermitian matrix

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

    UPMNS = np.copy(_U)

    HD = np.copy(DM)

    N = UPMNS @ HD
    T = np.copy(matrixAdjoint(UPMNS))
    A = N @ T

    Hff = A + Pot

    eigvalues, eigvectors = np.linalg.eigh(Hff)

    V = eigvectors

    S[0,0] = np.exp(-ProbConst.I * eigvalues[0] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[0,1] = DM[0][0]
    S[0,2] = DM[0][0]
    S[1,0] = DM[0][0]
    S[1,1] = np.exp(-ProbConst.I * eigvalues[1] * _L * 1e9 / ProbConst.GevkmToevsq)
    S[1,2] = DM[0][0]
    S[2,0] = DM[0][0]
    S[2,1] = DM[0][0]
    S[2,2] = np.exp(-ProbConst.I * eigvalues[2] * _L * 1e9 / ProbConst.GevkmToevsq)

    S = (V) @ S @ (np.linalg.inv(V))

    for i in range(3):
        for j in range(3):
            _P[i][j] = abs(S[:,i][j] * S[:,i][j])

def ProbabilityVis(_energy, _L, _rho, _th, _dm, d, _alpha, _mlight, _tfi, _tsi, _tff, _tsf, _tpar, _thij, _tqcoup):

    Pot = np.ndarray((3,3), dtype=np.complex)

    Enf = _energy * 1.e9
    mss = np.ndarray((3),  dtype=np.complex)
    
    if (_dm[1] > 0):
        mss[0] = _mlight
        mss[1] = _dm[0] + _mlight
        mss[2] = _dm[1] + _mlight
    else:
        mss[0] = _mlight
        mss[1] = -_dm[1] + _mlight
        mss[2] = _dm[0] - _dm[1] + _mlight
    
    mm = min(20, (mss[_tpar] / mss[_thij]) * _energy)

    Pot[0,0] = np.complex(_rho * 7.63247 * 0.5 * 1e-14, 0)
    Pot[0,1] = ProbConst.Z0
    Pot[0,2] = ProbConst.Z0
    Pot[1,0] = ProbConst.Z0
    Pot[1,1] = ProbConst.Z0
    Pot[1,2] = ProbConst.Z0
    Pot[2,0] = ProbConst.Z0
    Pot[2,1] = ProbConst.Z0
    Pot[2,2] = ProbConst.Z0
    return np.real(1.e9 * NintegrGLQ(lambda x: PhiIntegration(x, mss, Pot, Enf, _L, _th, d, _alpha, _tfi, _tsi, _tff, _tsf, _tpar, _thij, _tqcoup), (1.00001) * _energy, mm))