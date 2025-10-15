import sys

import numpy as np
from numpy import exp, log10, log, sin, cos
from math import gamma
from scipy.integrate import quad
from scipy.constants import pi
from scipy.optimize import brentq

from uncertainties.unumpy import nominal_values as val


quadlimit = 1000


def convert2iterable(a):
    try:
        iter(a)
        iterable = a
    except TypeError:
        iterable = [a]
    return iterable


def convert2noniterable(a):
    try:
        iter(a)
        return a[0]
    except TypeError:
        return a


def elementwise(fun, iterable):

    if len(convert2iterable(iterable)) == 1:
        return convert2noniterable(fun(iterable))

    else:
        result = []
        for i in convert2iterable(iterable):
            result.append(fun(i))
        return np.array(result)


def inverseKohlrausch(phi, tau, beta):
    time = tau * (-log(phi)) ** (1 / beta)
    return time


def tauKfromTauAve(tauAve, beta):
    try:
        return beta * tauAve / (gamma(1 / beta))
    except OverflowError:
        return np.inf


class TNMG(object):

    def __init__(self, myegalogFun, myegaCoeff, GinfFun, beta, T1, numXi):

        self.Ginf = GinfFun
        self.vfun = myegalogFun
        self.K = val(myegaCoeff["C"])
        self.tauinf = 10 ** (val(myegaCoeff["ninf"])) / self.Ginf(T1)
        self.init(beta, T1, numXi)

    def init(self, beta, T1, numXi, normalized=False):
        self.T1 = T1
        self.beta = beta
        self.numXi = numXi
        self.tauK = tauKfromTauAve(10 ** self.vfun(T1) / self.Ginf(T1), beta)
        self.xi = self.getxi(numXi)
        self.taui = self.gettaui(self.xi, self.tauK)
        self.gifun = self.getgifun(beta, self.tauK, normalized)
        self.gi = self.getgi(self.taui, self.gifun)
        self.Bi = self.getBi(self.taui, self.tauinf, self.T1, self.K)

    def Scherer_limits(self, numberOfValues):
        beta = self.beta
        lim1 = (0.0157 * exp(-7.93 * beta)) ** (1 / beta)
        lim2 = (10.34 - 10.14 * beta) ** (1 / beta)
        return np.logspace(log10(lim1), log10(lim2), numberOfValues)

    def getxi(self, numXi):
        xi = self.Scherer_limits(numXi)  # this is = taui/tauK
        return xi

    def getgifun(self, beta, tauK, normalized=False):

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
        else:
            return gi

    def gettaui(self, xi, tauK):
        return xi * tauK

    def getgi(self, taui, gifun):
        gi = []
        for tau in taui:
            gi.append(gifun(tau))
        gi = np.array(gi)
        gi = gi / np.sum(gi)  # a soma tem que dar 1
        return gi

    def getBi(self, taui, tauinf, T1, K):
        # Cada unidade de relaxação tem seu Bi (acho)
        return log(taui / tauinf) * T1 / exp(K / T1)

    def tauFun(self, T, i):
        # Vem do MYEGA
        return self.tauinf * exp((self.Bi[i] / T) * exp(self.K / T))

    def tauR(self, T):
        # acho que é o tau de relaxação

        def fun(T):
            # Acho que é o valor esperado de taui
            taui = []
            for i in range(len(self.gi)):
                taui.append(self.tauFun(T, i))
            return np.sum(self.gi * taui)

        return elementwise(fun, T)

    def tauD(self, T):
        # é a freq de relaxação? não lembro disso

        def fun(T):
            taui = []
            for i in range(len(self.gi)):
                taui.append(self.tauFun(T, i))
            return 1 / np.sum(self.gi / taui)

        return elementwise(fun, T)

    def time(self, i, T2, Tfi):
        # evolução isotérmica do tempo com relação à Tfi. Ver pdf aula 9 gupta pag 16.

        if Tfi == T2:
            return np.inf
        elif Tfi == self.T1:
            return 0

        def insideIntegral(Tfi):
            return exp((self.Bi[i] / T2) * exp(self.K / Tfi)) / (T2 - Tfi)

        integral = quad(insideIntegral, self.T1, Tfi)[0]

        return self.tauinf * integral

    def Tfi(self, i, T2, time):
        # evolução isotérmica de Tfi com relação ao tempo t. Lembrar que cada i
        # é uma estrutura.
        def inverse(T):
            return self.time(i, T2, T) - time

        return brentq(inverse, self.T1, T2)

    def TfH(self, T2, time):
        # Relaxação da entalpia (ou outra propriedade) que depende dos gis e dos
        # Tfis de cada estrutura
        result = []
        for i in range(len(self.gi)):
            gi = self.gi[i]
            Tfi = self.Tfi(i, T2, time)
            result.append(Tfi * gi)
        return np.sum(result)

    def phi(self, T2, time):
        # este é o phi do Kohlrausch
        phi = (self.TfH(T2, time) - T2) / (self.T1 - T2)
        return phi

    def timeToPhi(self, T2, phi, maxiter=100):
        # resolve o problema inverso de encontrar um tempo para uma certa relaxação

        beta = self.beta
        tauK_T2 = tauKfromTauAve(10 ** self.vfun(T2) / self.Ginf(T2), beta)

        time0 = 0
        time1 = inverseKohlrausch(phi, tauK_T2, beta)

        def fun(t):
            return self.phi(T2, t) - phi

        try:
            return brentq(fun, time0, time1, maxiter=maxiter)

        except ValueError:
            return brentq(fun, time0, time1 * 1e5)


if __name__ == "__main__":

    fullyRelaxed = 99  # percent
    phiRelaxed = 1 - fullyRelaxed / 100

    # Para a curva de menor viscosidade:

    A = -0.81273
    Tg = 802.3626
    m = 40.57333
    name = "menorVisco"

    def fun(T):
        return A + (12 - A) * (Tg / T) * exp(((m / (12 - A)) - 1) * ((Tg / T) - 1))

    def Ginf(T):
        return 29e9

    coeffs = {
        "C": Tg * (m / (12 - A) - 1),
        "ninf": A,
    }

    minT = Tg - 150
    maxT = Tg + 50

    minT = Tg - 300
    maxT = Tg - 150

    Trange_ = np.linspace(minT, maxT, 10)

    T1 = Tg
    beta = 0.5

    y = []

    vrelax = TNMG(fun, coeffs, Ginf, beta, T1, 10)

    for T2 in Trange_:
        time = vrelax.timeToPhi(T2, phiRelaxed)
        y.append(time)
        print(T2, time)
        sys.stdout.flush()
