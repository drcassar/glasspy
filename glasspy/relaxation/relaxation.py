import sys
import warnings
from functools import partial
from math import gamma

import numpy as np
from glasspy.viscosity.equilibrium_log import myega_alt as myega
from joblib import Parallel, delayed
from numpy import cos, exp, log, log10, sin
from scipy.constants import pi
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

quadlimit = 1000


def inv_kohlrausch(phi, tau, beta):
    time = tau * (-log(phi)) ** (1 / beta)
    return time


def tauk_from_tauave(tauAve, beta):
    try:
        return beta * tauAve / (gamma(1 / beta))
    except OverflowError:
        return np.inf


class TNMG(object):
    """Class for the Tool-Naranayaswamy-Moynihan-Gupta model.

    Args:
      myega_T12: Temperature where viscosity is 10**12 Pa.s.
      myega_m: Liquid fragility.
      myega_log_eta_inf: base-10 log of eta_infinity from MYEGA.
      Ginf_fun: Function of shear modulus at infinite frequency.
      beta: relaxation exponent.
      T1: initial temperature.
      num_xi: number of relaxation structures.
    """

    def __init__(
        self,
        myega_T12,
        myega_m,
        myega_log_eta_inf,
        Ginf_fun,
        beta,
        T1,
        num_xi,
    ):

        self.Ginf = Ginf_fun
        self.T1 = T1
        self.beta = beta
        self.num_xi = num_xi

        self.K = myega_T12 * (myega_m / (12 - myega_log_eta_inf) - 1)

        self.logvisc = partial(
            myega,
            log_eta_inf=myega_log_eta_inf,
            T12=myega_T12,
            m=myega_m,
        )

        self.tauinf = 10 ** (myega_log_eta_inf) / self.Ginf(T1)
        self.tauK = tauk_from_tauave(10 ** self.logvisc(T1) / self.Ginf(T1), beta)

        self.xi = self.getxi()
        self.taui = self.xi * self.tauK

        self.gifun = self.getgifun(normalized=False)
        self.gi = self.getgi()

        self.Bi = log(self.taui / self.tauinf) * self.T1 / exp(self.K / self.T1)

    def getxi(self):
        """This is related to the Scherer limits.

        xi = taui / tauK

        """
        lim1 = (0.0157 * exp(-7.93 * self.beta)) ** (1 / self.beta)
        lim2 = (10.34 - 10.14 * self.beta) ** (1 / self.beta)
        return np.logspace(log10(lim1), log10(lim2), self.num_xi)

    def getgifun(self, normalized=False):

        beta = self.beta
        tauK = self.tauK

        if int(1000 * beta) == int(500):

            def gi(tau):
                ln_u = log(tau / tauK)
                return exp(ln_u / 2) * exp(-exp(ln_u) / 4) / (2 * pi ** (1 / 2))

        else:

            if beta <= 0.51:

                def lambda_(x):
                    # Eq. (23) from lindsey and patterson

                    def intfun(u):
                        return (
                            exp(-x * u)
                            * exp(-(u**beta) * cos(pi * beta))
                            * sin((u**beta) * sin(pi * beta))
                        )

                    return (1 / pi) * quad(intfun, 0, np.inf, limit=quadlimit)[0]

            else:

                def lambda_(x):
                    # foi adaptado do artigo de berberan-santos

                    def intfun(u):
                        return exp((-(u**beta)) * cos(beta * pi / 2)) * cos(
                            (u**beta) * sin(beta * pi / 2) - x * u
                        )

                    return (1 / pi) * quad(intfun, 0, np.inf, limit=quadlimit)[0]

            def gi(tau):
                return (tauK / (tau**2)) * lambda_(tauK / tau)

        if normalized:
            norm = quad(gi, 0, np.inf, limit=quadlimit)[0]

            def ginorm(tau):
                return gi(tau) / norm

            return ginorm

        return gi

    def getgi(self):
        gi = []
        for tau in self.taui:
            gi.append(self.gifun(tau))
        gi = np.array(gi)
        gi = gi / np.sum(gi)  # normalizing
        return gi

    def tauFun(self, T):
        """From the MYEGA eq."""
        return self.tauinf * exp((self.Bi / T) * exp(self.K / T))

    def tauR(self, T):
        # acho que é o tau de relaxação

        T_arr = np.atleast_1d(T)
        taui_matrix = self.tauinf * exp(
            (self.Bi / T_arr[:, None]) * exp(self.K / T_arr[:, None])
        )
        result = np.sum(self.gi * taui_matrix, axis=1)

        return result.item() if np.isscalar(T) else result

    def tauD(self, T):

        T_arr = np.atleast_1d(T)
        taui_matrix = self.tauinf * exp(
            (self.Bi / T_arr[:, None]) * exp(self.K / T_arr[:, None])
        )
        result = 1 / np.sum(self.gi / taui_matrix, axis=1)

        return result.item() if np.isscalar(T) else result

    def time_single(self, i, T2, Tfi):
        # evolução isotérmica do tempo com relação à Tfi. Ver pdf aula 9 gupta pag 16.

        if Tfi == T2:
            return np.inf

        elif Tfi == self.T1:
            return 0

        def insideIntegral(Tfi_inner):
            return exp((self.Bi[i] / T2) * exp(self.K / Tfi_inner)) / (T2 - Tfi_inner)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            return self.tauinf * quad(insideIntegral, self.T1, Tfi, limit=quadlimit)[0]

    def Tfi_single(self, i, T2, time):
        # evolução isotérmica de Tfi com relação ao tempo t. Lembrar que cada i
        # é uma estrutura.
        def inverse(T):
            return self.time_single(i, T2, T) - time

        return brentq(inverse, self.T1, T2)

    def time(self, T2, Tfi, n_jobs=-1):
        i_arr = np.atleast_1d(range(len(self.gi)))

        result = Parallel(n_jobs=n_jobs)(
            delayed(self.time_single)(i_val, T2, Tfi) for i_val in i_arr
        )

        result = np.array(result)
        return result.item() if np.isscalar(result) else result

    def Tfi(self, T2, time, n_jobs=-1):
        i_arr = np.atleast_1d(range(len(self.gi)))

        result = Parallel(n_jobs=n_jobs)(
            delayed(self.Tfi_single)(i_val, T2, time) for i_val in i_arr
        )

        result = np.array(result)
        return result.item() if np.isscalar(result) else result

    def TfH(self, T2, time, n_jobs=-1):
        # Relaxação da entalpia (ou outra propriedade) que depende dos gis e dos
        # Tfis de cada estrutura
        Tfi = self.Tfi(T2, time, n_jobs)
        return np.sum(Tfi * self.gi)

    def phi(self, T2, time, n_jobs=-1):
        # este é o phi do Kohlrausch
        phi = (self.TfH(T2, time, n_jobs=-1) - T2) / (self.T1 - T2)
        return phi

    def timeToPhi(self, T2, phi, maxiter=100):
        # resolve o problema inverso de encontrar um tempo para uma certa relaxação

        tauK_T2 = tauk_from_tauave(10 ** self.logvisc(T2) / self.Ginf(T2), beta)

        time0 = 0
        time1 = inv_kohlrausch(phi, tauK_T2, self.beta)

        def fun(t):
            return self.phi(T2, t) - phi

        try:
            return brentq(fun, time0, time1, maxiter=maxiter)

        except ValueError:
            return brentq(fun, time0, time1 * 1e5)


if __name__ == "__main__":

    fullyRelaxed = 99  # percent
    phiRelaxed = 1 - fullyRelaxed / 100

    A = -0.81273
    Tg = 802.3626
    m = 40.57333

    def Ginf(T):
        return 29e9

    minT = Tg - 150
    maxT = Tg + 50

    minT = Tg - 300
    maxT = Tg - 150

    Trange_ = np.linspace(minT, maxT, 10)

    T1 = Tg
    beta = 0.5

    y = []

    model = TNMG(Tg, m, A, Ginf, beta, T1, 10)

    for T2 in Trange_:
        time = model.timeToPhi(T2, phiRelaxed)
        y.append(time)
        print(T2, time)
        sys.stdout.flush()
