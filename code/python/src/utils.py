from os import TMP_MAX
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

class ProbConst:

    I = np.complex(0,1)
    Z0 = np.complex(0,0)
    hbar = 6.58211928*1.e-25
    clight = 299792458
    GevkmToevsq = 0.197327

def square(x):
    return x * x

def make_umns(_sg, _th, _dcp):
    
    th12 = _th[0]
    th13 = _th[1]
    th23 = _th[2]
    delta = _sg * _dcp

    U = np.ndarray((3,3),dtype=np.complex)
    
    U[0,0] = np.cos(th12) * np.cos(th13)
    U[0,1] = np.sin(th12) * np.cos(th13)
    U[0,2] = np.sin(th13) * np.exp(-ProbConst.I * delta)
    U[1,0] = -np.sin(th12) * np.cos(th23) - np.cos(th12) * np.sin(th23) * np.sin(th13) * np.exp(ProbConst.I * delta)
    U[1,1] = np.cos(th12) * np.cos(th23) - np.sin(th12) * np.sin(th23) * np.sin(th13) * np.exp(ProbConst.I * delta)
    U[1,2] = np.sin(th23) * np.cos(th13)
    U[2,0] = np.sin(th12) * np.sin(th23) - np.cos(th12) * np.cos(th23) * np.sin(th13) * np.exp(ProbConst.I * delta)
    U[2,1] = -np.cos(th12) * np.sin(th23) - np.sin(th12) * np.cos(th23) * np.sin(th13) * np.exp(ProbConst.I * delta)
    U[2,2] = np.cos(th23) * np.cos(th13)
    
    return U

def dGdE(mi, mf, Ei, Ef, coup):
    
    xif = mi / mf
    ei = Ei *1.e-9
    ef = Ef *1.e-9
    tmp = 0.0
    
    if coup == 1:
        if ei / xif <= ef and ef <= ei:
            tmp = (1.0 / np.sqrt(1 - mi / pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * (1. / (ei * ef)) * (pow(ei + np.sqrt(xif) * ef, 2) / pow(np.sqrt(xif) + 1, 2))
    else:
        if ei / xif <= ef and ef <= ei:
            tmp = (1.0 / np.sqrt(1 - mi / pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * (1. / (ei * ef)) * (pow(ei - np.sqrt(xif) * ef, 2) / pow(np.sqrt(xif) - 1, 2))
    
    return tmp

def dGbdE(mi, mf, Ei, Ef, coup):
    
    xif = mi / mf
    ei = Ei *1.e-9
    ef = Ef *1.e-9
    tmp = 0.0
    
    if coup == 1:
        if ei / xif <= ef and ef <= ei:
            tmp = (1.0 / np.sqrt(1 - mi / pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * ((ei - ef) / (ei * ef)) * ((xif * ef - ei) / pow(np.sqrt(xif) + 1, 2))
    else:
        if ei / xif <= ef and ef <= ei:
            tmp = (1.0 / np.sqrt(1 - mi / pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * ((ei - ef) / (ei * ef)) * ((xif * ef - ei) / pow(np.sqrt(xif) - 1, 2))
    
    return tmp

def PhiIntegration(_ei, _mss, _Pot, _enf, _l, _th, _d, _alpha, _tfi, _tsi, _tff, _tsf, _tpar, _thij, _tqcoup):
    
    Eni = _ei * 1.e9
    ef = _enf * 1.e-9
    dist = _l * 1.e9 / ProbConst.GevkmToevsq

    Utemp = np.ndarray((3,3), dtype=np.complex)
    Htemp = np.ndarray((3,3), dtype=np.complex)

    _U = make_umns(_tsi, _th, _d)
    
    Utemp = np.copy(_U)
    
    Htemp[0,0] = np.complex(0.5 * _mss[0] / Eni, -0.5 * _alpha[0] / Eni)
    Htemp[0,1] = ProbConst.Z0
    Htemp[0,2] = ProbConst.Z0
    Htemp[1,0] = ProbConst.Z0
    Htemp[1,1] = np.complex(0.5 * _mss[1] / Eni, -0.5 * _alpha[1] / Eni)
    Htemp[1,2] = ProbConst.Z0
    Htemp[2,0] = ProbConst.Z0
    Htemp[2,1] = ProbConst.Z0
    Htemp[2,2] = np.complex(0.5 * _mss[2] / Eni, -0.5 * _alpha[2] / Eni)

    temp = Utemp @ Htemp @ np.copy(matrixAdjoint(Utemp)) + (_tsi * _Pot)

    eigenvalues, eigenvectors = sp.linalg.eig(temp) # no hermitian matrix

    massfisq = [2 * Eni * np.real(eigenvalues[0]), 2 * Eni * np.real(eigenvalues[1]), 2 * Eni * np.real(eigenvalues[2])]
    alphafi = [-2 * Eni * np.imag(eigenvalues[0]), -2 * Eni * np.imag(eigenvalues[1]), -2 * Eni * np.imag(eigenvalues[2])]

    Umati = eigenvectors

    Umatinvi = np.linalg.inv(Umati)

    Cmati = Umati.transpose() @ Utemp.conj()


    _U = make_umns(_tsf, _th, _d)

    Utemp = np.copy(_U)
    
    Htemp[0,0] = np.complex(0.5 * _mss[0] / _enf, -0.5 * _alpha[0] / _enf)
    Htemp[0,1] = ProbConst.Z0
    Htemp[0,2] = ProbConst.Z0
    Htemp[1,0] = ProbConst.Z0
    Htemp[1,1] = np.complex(0.5 * _mss[1] / _enf, -0.5 * _alpha[1] / _enf)
    Htemp[1,2] = ProbConst.Z0
    Htemp[2,0] = ProbConst.Z0
    Htemp[2,1] = ProbConst.Z0
    Htemp[2,2] = np.complex(0.5 * _mss[2] / _enf, -0.5 * _alpha[2] / _enf)

    temp = (Utemp @ Htemp @ np.copy(matrixAdjoint(Utemp))) + (_tsf * _Pot)

    eigenvalues, eigenvectors = np.linalg.eig(temp)
    massffsq = [2 * _enf * np.real(eigenvalues[0]), 2 * _enf * np.real(eigenvalues[1]), 2 * _enf * np.real(eigenvalues[2])]
    alphaff = [-2 * _enf * np.imag(eigenvalues[0]), -2 * _enf * np.imag(eigenvalues[1]), -2 * _enf * np.imag(eigenvalues[2])]

    Umatf = eigenvectors

    Cmatinvf = Umatf.transpose() @ Utemp.conj()
    Cmatinvf = np.linalg.inv(Cmatinvf)

    theta = 0

    if (_tsi * _tsf > 0):
        theta = dGdE(_mss[_tpar], _mss[_thij], Eni, _enf, _tqcoup)
    else:
        theta = dGbdE(_mss[_tpar], _mss[_thij], Eni, _enf, _tqcoup)

    sum = 0.0
    tmp = 0.0
    for i in range(3):
        for p in range(3):
            for h in range(3):
                for n in range(3):
                    sum = sum + np.real(Umatinvi[:,_tfi][i] * np.conj(Umatinvi[:,_tfi][p]) * Umatf[:,h][_tff] * np.conj(Umatf[:,n][_tff]) * (((ef / _ei) * (alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n]) - ProbConst.I * ((ef / _ei) * ((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n])))) / (square((ef / _ei) * (alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n])) + square((ef / _ei) * ((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n]))))) * (np.exp(-ProbConst.I * ((massffsq[h]) - (massffsq[n])) / (2 * _enf) * dist) * np.exp(-((alphaff[h] + alphaff[n]) / (2 * _enf)) * dist) - np.exp(-ProbConst.I * ((massfisq[i]) - (massfisq[p])) / (2 * Eni) * dist) * np.exp(-((alphafi[i] + alphafi[p]) / (2 * Eni)) * dist)) * Cmatinvf[:,h][_thij] * np.conj(Cmatinvf[:,n][_thij]) * Cmati[:,_tpar][i] * np.conj(Cmati[:,_tpar][p]))
    
    tmp = 2 * sum * (((ef / _ei) * _alpha[_tpar]) / Eni) * theta
    return tmp

def matrixAdjoint(matrix):
 
    try:
        determinant = np.linalg.det(matrix)
        if(determinant!=0):
            adj = None
            adj = np.linalg.inv(matrix) * determinant
            # return cofactor matrix of the given matrix
            return adj
        else:
            raise Exception("singular matrix")
    except Exception as e:
        print("could not find adjoint matrix due to",e)