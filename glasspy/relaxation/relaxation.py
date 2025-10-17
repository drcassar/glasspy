import warnings
from functools import partial
from math import cos
from math import exp as exp_
from math import factorial, gamma, sin

import numpy as np
from mpmath import inf, nsum
from numpy import exp, log, log10
from scipy.constants import pi
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

from glasspy.viscosity.equilibrium_log import myega_alt as myega

quadlimit = 1000


def inv_kohlrausch(phi, tau, beta):
    time = tau * (-log(phi)) ** (1 / beta)
    return time


def tauk_from_tauave(tauAve, beta):
    try:
        return tauAve / (gamma(1 / beta + 1))
    except OverflowError:
        return np.inf


def lambda_lp_22(x, beta=1 / 2):
    """Eq. (22) from Lindsey and Patterson

    input x is tauK / tau. Only works for beta=1/2.

    Ref:
      C.P. Lindsey, G.D. Patterson, Detailed comparison of the Williams–Watts
      and Cole–Davidson functions, The Journal of Chemical Physics 73 (1980)
      3348.
    """

    return (1 / 2) * pi ** (-1 / 2) * x ** (-3 / 2) * exp(-1 / (4 * x))


def lambda_lp_23(x, beta):
    """Eq. (23) from Lindsey and Patterson

    input x is tauK / tau

    Ref:
      C.P. Lindsey, G.D. Patterson, Detailed comparison of the Williams–Watts
      and Cole–Davidson functions, The Journal of Chemical Physics 73 (1980)
      3348.
    """

    def intfun(u):
        return (
            exp_(-x * u)
            * exp_(-(u**beta) * cos(pi * beta))
            * sin((u**beta) * sin(pi * beta))
        )

    return (1 / pi) * quad(intfun, 0, np.inf, limit=quadlimit)[0]


def lambda_lp_24(x, beta):
    """Eq. (24) from Lindsey and Patterson

    input x is tauK / tau

    Ref:
      C.P. Lindsey, G.D. Patterson, Detailed comparison of the Williams–Watts
      and Cole–Davidson functions, The Journal of Chemical Physics 73 (1980)
      3348.
    """

    def sumfun(k):
        return (
            (-1) ** k
            / factorial(int(k))
            * sin(pi * beta * k)
            * gamma(beta * k + 1)
            / x ** (beta * k + 1)
        )

    return -(1 / pi) * float(nsum(sumfun, [0, inf]))


def lambda_bs(x, beta):
    """From Berberan-Santos

    Ref:
      M.N. Berberan-Santos, E.N. Bodunov, B. Valeur, Mathematical functions for
      the analysis of luminescence decays with underlying distributions 1.
      Kohlrausch decay function (stretched exponential), Chemical Physics 315
      (2005) 171–182.
    """

    def intfun(u):
        return exp_((-(u**beta)) * cos(beta * pi / 2)) * cos(
            (u**beta) * sin(beta * pi / 2) - x * u
        )

    return (1 / pi) * quad(intfun, 0, np.inf, limit=quadlimit)[0]


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

        self.gifun = self.getgifun()
        self.gi = self.getgi()

        self.Bi = log(self.taui / self.tauinf) * self.T1 / exp(self.K / self.T1)

    def getxi(self):
        """This is related to the Scherer limits.

        xi = taui / tauK

        """
        lim1 = (0.0157 * exp(-7.93 * self.beta)) ** (1 / self.beta)
        lim2 = (10.34 - 10.14 * self.beta) ** (1 / self.beta)
        return np.logspace(log10(lim1), log10(lim2), self.num_xi)

    def getgifun(self):

        def gi(tau):
            return (self.tauK / (tau**2)) * lambda_lp_24(self.tauK / tau, self.beta)

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

    def Tfi(self, T2, time):
        return np.array([self.Tfi_single(i, T2, time) for i in range(len(self.gi))])

    def TfH(self, T2, time, n_jobs=-1):
        # Relaxação da entalpia (ou outra propriedade) que depende dos gis e dos
        # Tfis de cada estrutura
        return np.sum(self.Tfi(T2, time) * self.gi)

    def phi(self, T2, time):
        # este é o phi do Kohlrausch
        phi = (self.TfH(T2, time) - T2) / (self.T1 - T2)
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

    # Experiment 1 - time to relax x percent

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

    # for T2 in Trange_:
    #     time = model.timeToPhi(T2, phiRelaxed)
    #     y.append(time)
    #     print(T2, time)
    #     sys.stdout.flush()

    # ---

    # Experiment 2 - compute phi from data

    time = [0, 600, 1800, 4200, 9600, 20400, 42000, 85200, 171600, 344400, 689976]
    phi = [1.000, 0.671, 0.529, 0.376, 0.259, 0.153, 0.059, 0.035, 0.000, 0.000, 0.000]

    Tg = 867.71239
    m = 37.3591
    log_eta_inf = -1.30458

    T1 = 868.15
    T2 = 828.15

    def Ginf(T):
        return 30.02e9

    beta = 0.90647165
    num_xi = 10

    model = TNMG(Tg, m, A, Ginf, beta, T1, num_xi)

    phi_calc = []

    for t in time:
        phi_calc.append(model.phi(T2, t))

    print(np.round(phi_calc, 3))
    print(np.array(phi))

    # ---

    # Experiment 3 - beta regression

    from scipy.optimize import curve_fit

    def phi_calc(x, beta):
        model = TNMG(Tg, m, A, Ginf, beta, T1, num_xi)
        phi = [model.phi(T2, t) for t in x]
        return phi

    x0 = [0.5]

    res, _ = curve_fit(
        phi_calc,
        time,
        phi,
        x0,
        bounds=((0.05, 1)),
    )

    print(res)
