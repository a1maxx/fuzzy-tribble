import numpy as np
from matplotlib import pyplot as plt
from matplotlib import figure
import math
from math import log10, sqrt, log2
import cmath


def db2pow(d):
    return 10 ** (d / 10)


N = 10 ** 5
Pt = 30
pt = 10 ** -3 * db2pow(Pt)
No = -114
no = 10 ** -3 * db2pow(No)

r = np.arange(0.5, 10.5, 0.5)

df, dn, eta = 1000, 500, 4

p1 = np.zeros(shape=(len(r),))
p2 = np.zeros(shape=(len(r),))
pa1 = np.zeros(shape=(len(r),))
pa2 = np.zeros(shape=(len(r),))

af, an = 0.75, 0.25
hf = sqrt(df ** -eta) * (np.random.normal(size=(N,)) + 1j * np.random.normal(size=(N,))) / sqrt(2)

hf = np.array(hf).astype(dtype=np.complex_)

hn = sqrt(dn ** -eta) * (np.random.normal(size=(N,)) + 1j * np.random.normal(size=(N,))) / sqrt(2)

hn = np.array(hn).astype(dtype=np.complex_)
abs(hf.real) ** 2 + hf.imag ** 2
g1 = np.abs(hf) ** 2
g2 = np.abs(hn) ** 2

for u in range(0, len(r)):
    epsilon = (2 ** (r[u])) - 1
    aaf = np.minimum(1, epsilon * (no + pt * g1) / (pt * g1 * (1 + epsilon)))
    aan = 1 - aaf

    gamma_f = pt * af * g1 / (pt * g1 * an + no)
    gamma_nf = pt * af * g2 / (pt * g2 * an + no)
    gamma_n = pt * g2 * an / no

    gamm_f = pt * aaf * g1 / (pt * g1 * aan + no)
    gamm_nf = pt * aaf * g2 / (pt * g2 * aan + no)
    gamm_n = pt * g2 * aan / no

    Cf = np.log2(1 + gamma_f)
    Cnf = np.log2(1 + gamma_nf)
    Cn = np.log2(1 + gamma_n)

    Ca_f = np.log2(1 + gamm_f)
    Ca_nf = np.log2(1 + gamm_nf)
    Ca_n = np.log2(1 + gamm_n)

    for k in range(0, N):
        if Cf[k] < r[u]:
            p1[u] += 1
        if Cnf[k] < r[u] or Cn[k] < r[u]:
            p2[u] += 1
        if Ca_f[k] < r[u]:
            pa1[u] += 1
        if aaf[k] != 0:
            if Ca_n[k] < r[u] or Ca_nf[k] < r[u]:
                pa2[u] += 1
        else:
            if Ca_n[k] < r[u]:
                pa2[u] += 1

pout1 = p1 / N
pout2 = p2 / N
pouta1 = pa1 / N
pouta2 = pa2 / N

plt.clf()
fig = plt.figure(figsize=(4.5, 3), dpi=300)

plt.plot(r, pout1, label="Far User (Fixed)", color="red", ls=":")
plt.plot(r, pout2, label="Near User (Fixed)", color="red", ls="-.")
plt.plot(r, pouta1, label="Far User (Dynamic)", color="green", ls="-")
plt.plot(r, pouta2, label="Near User (Dynamic)", color="green", ls="--")
plt.xlabel("Transmission Rate Requirements (bps/Hz)", size="large", weight="bold")
plt.ylabel("Probability of Outage", size="large", weight="bold")
plt.legend()

plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets

states = {1: (7, 3.5), 2: (8, 2.5), 3: (7, 3.25), 4: (8, 5)}

from scipy.stats import weibull_min

for i in states.keys():
    n = 500  # number of samples
    k = states[i][0]  # shape
    lam = states[i][1]  # scale
    x = weibull_min.rvs(k, loc=0, scale=lam, size=n)
    sns.kdeplot(x, shade=False, linewidth=3, color="orange")

plt.show()
plt.clf()

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 600
import numpy as np
from scipy.stats import gaussian_kde


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors]


data = x
# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
density = gaussian_kde(data)
xs = np.linspace(min(data), max(data), 200)
density.covariance_factor = lambda: .25
density._compute_covariance()
# plt.plot(xs, density(xs))

plt.plot(xs, density(xs))
plt.fill_between(xs[1:100], density(xs)[1:100], 0, alpha=0.5)
plt.fill_between(xs[99:120], density(xs)[99:120], 0, alpha=0.5, color="orange")

color1 = '#FB575D'
color2 = '#15251B'
plt.fill_between(xs[119:150], density(xs)[119:150], 0, alpha=0.5,
                 color=get_color_gradient(color1, color2, len(xs[119:150])))

plt.fill_between(xs[149:170], density(xs)[149:170], 0, alpha=0.5,
                 color="#be2596")

plt.fill_between(xs[169:len(xs)], density(xs)[169:len(xs)], 0, alpha=0.5,
                 color="#9925be")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Density")

plt.show()
