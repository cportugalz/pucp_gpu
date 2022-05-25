import numpy as np
import utils
import probabilities as prob
import os

num_simulations = 10

# Standard Oscilation
d = -1.57
L = 1300
rho = 2.956740
s = 1
a = 1
b = 0
th = [ 0.59, 0.15, 0.84 ]
dm = [ 7.4e-5, 2.5e-3 ]
alpSTD = np.zeros(3, dtype=np.complex)

# Invisible Decay
alpINV = [ 0, 0, 5.e-5 ]

# VEP
alpVEP = [ 0, 4.e-24, 0 ]

# Non Standard Interaction
#   ee, mm,tt pertenecen a Reales
#   em, emf -> modulo y fase
#   et, etf -> modulo y fase 
#   mt, mtf -> "           "
ee = 0
mm = 0
tt = 0
em = 0.05
emf = -1.55
et = 0
etf = 0
mt = 0
mtf = 0
alpNSI = [ ee, mm, tt, em, emf, et, etf, mt, mtf ]
delta = s * d

# Visible decay?
fi_1 = 1
si_1 = 1
fi_2 = 0
si_2 = 1
ff_1 = 0
sf_1 = 1
ff_2 = 1
sf_2 = -1
par = 2
hij = 0
qcoup = 1
mlight = 0.05 * 0.05

PrSTD = np.ndarray((3,3))
PrINV = np.ndarray((3,3))
PrVEP = np.ndarray((3,3))
PrNSI = np.ndarray((3,3))
PrDCH = np.ndarray((3,3))

U1 = utils.make_umns(s, th, d)

if not os.path.exists('output/'):
    os.mkdir("output")

file = open("output/output.txt", "w")

iter_energy = 1
while (iter_energy <= num_simulations):
    energy = iter_energy / 100.0
    prob.StandardOscilation(U1, energy, s, L, rho, dm, alpSTD, PrSTD)
    prob.InvisibleDecay(U1, energy, s, L, rho, dm, alpINV, PrINV)
    prob.ViolationPrincipleDecay(U1, energy, s, L, rho, dm, alpVEP, PrVEP)
    file.write(f'{energy:.2f},{PrSTD[1][0]:.8f},{PrINV[1][0]:.8f},{PrVEP[1][0]:.8f}\n')
    iter_energy += 1

file.close()