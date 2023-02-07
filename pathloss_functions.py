import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]


def PL_free(d: float, fc: float = 1500e6, Gt: float = 1, Gr: float = 1):
    """
    Free Space path loss model
    :fc : Carrier frequency
    :d : Distance between Tx and Rx (in m)
    :Gt/ Gr : antenna gains

    Out: PL: pathloss (dB)
    """
    lam = 3e8 / fc
    PL = (lam * np.sqrt(Gr * Gt)) / (4 * np.pi * d)
    return -20 * np.log10(PL)


d = np.linspace(1, 10 ** 3, num=10000)
GtGr_ll = [(1, 1), (1, 0.5), (0.5, 0.5)]

for GtGr in GtGr_ll:
    Gt, Gr = GtGr
    plt.semilogx(d, PL_free(d,Gt=Gt,Gr=Gr))

legend = [f"Gt {GtGr[0]} Gr {GtGr[1]}" for GtGr in GtGr_ll]

plt.xlabel("Distance [m]")
plt.ylabel("Path Loss [dB]")
plt.legend(legend)
plt.grid()
plt.show()


def PL_Hata(
    d: float, fc: float = 1500e6, hrx: float = 2, htx: float = 30, env: str = "Urban"
):
    """
    Hata path loss model, for large scale coverage
    :fc : Carrier frequency
    :d : Distance between Tx and Rx (in m)
    :hrx: height of receiving antenna
    :htx : height of transmitting antenna
    :env: Environment

    Out: PL: pathloss (dB)
    """
    assert env in ["Urban", "Suburban", "Rural"], f"Environment {env} not defined."
    fc /= 1e6

    # Low
    if (fc >= 150) and (fc <= 200):
        C_rx = 8.29 * (np.log10(1.54 * hrx)) ** 2 - 1.1
    elif fc >= 200:
        C_rx = 3.2 * (np.log10(11.75 * hrx)) ** 2 - 4.97
    else:
        C_rx = 0.8 + (1.1 * np.log10(fc) - 0.7) * hrx - 1.56 * np.log10(fc)

    PL = (
        69.55
        + 26.16 * np.log10(fc)
        - 13.82 * np.log10(htx)
        - C_rx
        + (44.9 - 6.55 * np.log10(htx)) * np.log10(d /1000)
    )

    if env == "Suburban":
        PL -= 2 * ((np.log10(fc / 28)) ** 2) - 5.4
    elif env == "Rural":
        PL += (18.33 - 4.78 * np.log10(fc)) * np.log10(fc) - 40.97
    return PL


d = np.linspace(10, 10 ** 5, num=10000)

# environments
legend = ["Urban","Suburban","Rural"]

for env in legend:
    plt.semilogx(d, PL_Hata(d,env=env))

plt.xlabel("Distance [m]")
plt.ylabel("Path Loss [dB]")
plt.title("Hata PathLoss model")
plt.legend(legend)
plt.grid()
plt.show()



