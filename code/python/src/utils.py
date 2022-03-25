from os import TMP_MAX
import numpy as np
import scipy as sp

class ProbConst:

    I = np.complex(0,1)
    Z0 = np.complex(0,0)
    hbar = 6.58211928*1.e-25
    clight = 299792458
    GevkmToevsq = 0.197327

def make_umns(_sg, _th, _dcp):
    
    th12 = _th[0]
    th13 = _th[1]
    th23 = _th[2]
    delta = _sg * _dcp

    U = np.array((3,3),dtype=np.complex)
    
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
            tmp = (1.0 / np.sqrt(1 - mi / np.pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * (1. / (ei * ef)) * (np.pow(ei + np.sqrt(xif) * ef) / np.pow(np.sqrt(xif) + 1, 2))
    else:
        if ei / xif <= ef and ef <= ei:
            tmp = (1.0 / np.sqrt(1 - mi / np.pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * (1. / (ei * ef)) * (np.pow(ei - np.sqrt(xif) * ef) / np.pow(np.sqrt(xif) - 1, 2))
    
    return tmp

def dGbdE(mi, mf, Ei, Ef, coup):
    
    xif = mi / mf
    ei = Ei *1.e-9
    ef = Ef *1.e-9
    tmp = 0.0
    
    if coup == 1:
        if ei / xif <= ef and ef <= ei:
            tmp = (1.0 / np.sqrt(1 - mi / np.pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * ((ei - ef) / (ei * ef)) * ((xif * ef - ei) / np.pow(np.sqrt(xif) + 1, 2))
    else:
        if ei / xif <= ef and ef <= ei:
            tmp = (1.0 / np.sqrt(1 - mi / np.pow(ei * 1.e9, 2))) * (xif / (xif - 1)) * ((ei - ef) / (ei * ef)) * ((xif * ef - ei) / np.pow(np.sqrt(xif) - 1, 2))
    
    return tmp

def PhiIntegration(_ei, _mss, _Pot, _enf, _l, _th, _d, _alpha, _tfi, _tsi, _tff, _tsf, _tpar, _thij, _tqcoup):
    
    Eni = _ei * 1.e-9
    ef = _enf * 1.e-9
    dist = _l * 1.e-9 / ProbConst.GevkmToevsq

    Utemp = np.array((3,3))
    Htemp = np.array((3,3))

    _U = make_umns(_tsi, _th, _d)
    
    Utemp << _U[0,0], _U[0,1], _U[0,2], _U[1,0], _U[1,1], _U[1,2], _U[2,0], _U[2,1], _U[2,2]
    Htemp << np.complex(0.5 * _mss[0] / Eni, -0.5 * _alpha[0] / Eni), ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, np.complex(0.5 * _mss[1] / Eni, -0.5 * _alpha[1] / Eni), ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, np.complex(0.5 * _mss[2] / Eni, -0.5 * _alpha[2] / Eni)

    Htemp = Utemp * Htemp * Utemp.conj().T + _tsi + _Pot
    tmpVD = sp.linalg.eig(Htemp)

    massfisq = [2 * Eni * np.real(tmpVD[0][0]), 2 * Eni * np.real(tmpVD[0][1]), 2 * Eni * np.real(tmpVD[0][2])]
    alphafi = [-2 * Eni * np.imag(tmpVD[0][0]), -2 * Eni * np.imag(tmpVD[0][1]), -2 * Eni * np.imag(tmpVD[0][2])]

    Umati = tmpVD[1]
    Umatinvi = np.linalg.inv(Umati)

    Cmati = Umati.transpose * Utemp.conj()

    _U = make_umns(_tsf, _th, _d)
    Utemp << _U[0,0], _U[0,1], _U[0,2], _U[1,0], _U[1,1], _U[1,2], _U[2,0], _U[2,1], _U[2,2]
    Htemp << np.complex(0.5 * _mss[0] / _enf, -0.5 * _alpha[0] / _enf), ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, np.complex(0.5 * _mss[1] / _enf, -0.5 * _alpha[1] / _enf), ProbConst.Z0, ProbConst.Z0, ProbConst.Z0, np.complex(0.5 * _mss[2] / _enf, -0.5 * _alpha[2] / _enf)

    Htemp = Utemp * Htemp * Utemp.conj().T + _tsf + _Pot

    tmpVD = sp.linalg.eig(Htemp)

    massffsq = [2 * Eni * np.real(tmpVD[0][0]), 2 * Eni * np.real(tmpVD[0][1]), 2 * Eni * np.real(tmpVD[0][2])]
    alphaff = [-2 * Eni * np.imag(tmpVD[0][0]), -2 * Eni * np.imag(tmpVD[0][1]), -2 * Eni * np.imag(tmpVD[0][2])]

    Umatf = tmpVD[1]

    Cmatinvf = Umatf.transpose * Utemp.conj()
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
                    sum += np.real(Umatinvi[:,_tfi][i] * np.conj(Umatinvi[:,_tfi][p]) * Umatf[:,h][_tff] * np.conj(Umatf[:,n][_tff]) * (((ef / _ei) * (alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n]) - ProbConst.I * ((ef / _ei) * ((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n])))) / (np.pow((ef / _ei) * (alphafi[i] + alphafi[p]) - (alphaff[h] + alphaff[n]),2) + np.pow((ef / _ei) * ((massfisq[i]) - (massfisq[p])) - ((massffsq[h]) - (massffsq[n])),2))) * (np.exp(-ProbConst.I * ((massffsq[h]) - (massffsq[n])) / (2 * _enf) * dist) * np.exp(-((alphaff[h] + alphaff[n]) / (2 / _enf)) * dist) - np.exp(-ProbConst.T * ((massfisq[i]) - (massfisq[p])) / (2 * Eni) * dist) * np.exp(-((alphafi[i] + alphafi[p]) / (2 * Eni)) * dist)) * Cmatinvf[:,h][_thij] * np.conj(Cmatinvf[:,n][_thij]) * Cmati[:,_tpar][i] * np.conj(Cmati[:,_tpar][p]))
    tmp = 2 * sum * (((ef / _ei) * _alpha[_tpar]) / Eni) * theta
    return tmp

