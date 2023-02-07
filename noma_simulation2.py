import numpy as np
from matplotlib import pyplot as plt
from matplotlib import figure
import math
from math import log10, sqrt, log2
import cmath


def db2pow(d):
    return 10 ** (d / 10)


df, dn, eta, N = 800, 200, 4, 10 ** 5

Pt = np.arange(-60, 65, 5)
pt = (10 ** -3) * db2pow(Pt)

BW = 10 ** 6

No = -174 + 10 * np.log10(BW)
no = (10 ** -3) * db2pow(No)

rf, rn = 0.5, 0.5

## Rayleigh Fading Coefficients
hf = sqrt(df ** -eta) * (np.random.normal(size=(N,)) + 1j * np.random.normal(size=(N,))) / sqrt(2)
hf = np.array(hf).astype(dtype=np.complex_)
hn = sqrt(dn ** -eta) * (np.random.normal(size=(N,)) + 1j * np.random.normal(size=(N,))) / sqrt(2)
hn = np.array(hn).astype(dtype=np.complex_)
gf = np.abs(hf) ** 2
gn = np.abs(hn) ** 2

pf = np.zeros(shape=(len(pt),))
pn = np.zeros(shape=(len(pt),))
Rf = np.zeros(shape=(len(pt),))
Rn = np.zeros(shape=(len(pt),))

for i in range(0, len(pt)):
    Cf = np.log2(1 + gf * pt[i] / no)
    Cn = np.log2(1 + gn * pt[i] / (gf * pt[i] + no))

    for k in range(0,N):
        if Cf[k] < rf or Cn[k] < rn:
            pf[i] += 1
        if Cn[k] < rn:
            pn[i] += 1



plt.clf()
fig = plt.figure(figsize=(4.5, 3), dpi=300)


plt.semilogy(Pt, pf/N, label="Far User")
plt.semilogy(Pt, pn/N, label="Near User")
plt.legend()
plt.show()
