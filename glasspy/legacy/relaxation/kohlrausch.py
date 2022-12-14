'''Functions related to the Kohlrausch equation for relaxation, also known as
stretched exponential.'''

from math import gamma
from numpy import exp


def kohlrausch(time, tau_k, beta):
    '''Stretched exponential function commonly used for relaxation studies

    Parameters
    ----------
    time : float or array_like
        Elapsed time. The unity must be the same as tau_k.

    tau_k : float
        Characteristic time for the relaxation process. The unity must be the
        same as time.

    beta : float
        Stretched exponent. Must be greater than zero. Usually less than or
        equal to 1.

    Returns
    -------
    relaxation_parameter : float or array_like
        Number between zero and one indicating the degree of relaxation of the
        system. A value of one is for a system which is not relaxed. A value of
        zero is for a system fully relaxed.

    References
    ----------
    [1] Kohlrausch, R.H.A. (1854). Theorie des elektrischen Rückstandes in der
        Leidener Flasche. Annalen Der Physik Und Chemie 167, 179–214.

    [2] Ngai, K.L. (2011). Relaxation and Diffusion in Complex Systems (New
        York: Springer).

    '''
    relaxation_parameter = exp(-(time/tau_k)**beta)
    return relaxation_parameter


def tau_k2tau_ave(tau_k, beta):
    '''Converts the characteristic Kohlrausch time to the average relax. time

    Parameters
    ----------
    tau_k : float
        Characteristic time for the relaxation process.

    beta : float
        Stretched exponent. Must be greater than zero. Usually less than or
        equal to 1.

    Returns
    -------
    tau_ave : float
        Average relaxation time.

    '''
    tau_ave = tau_k*gamma(1/beta + 1)
    return tau_ave


def tau_ave2tau_k(tau_ave, beta):
    '''Converts the average relax. time to the characteristic Kohlrausch time

    Parameters
    ----------
    tau_ave : float
        Average relaxation time.

    beta : float
        Stretched exponent. Must be greater than zero. Usually less than or
        equal to 1.

    Returns
    -------
    tau_k : float
        Characteristic time for the relaxation process. 
    '''
    tau_k = tauAve/(gamma(1/beta + 1))
    return tau_k


def propertyRelaxation(time, tau_k, beta, p_0, p_inf):
    '''Computes the time-dependent property relaxation kinetics.

    Parameters
    ----------
    time : float or array_like
        Elapsed time. The unity must be the same as tau_k.

    tau_k : float
        Characteristic time for the relaxation process. The unity must be the
        same as time.

    beta : float
        Stretched exponent. Must be greater than zero. Usually less than or
        equal to 1.

    p_0 : float 
        Value of a certain property of the system at time equal zero.
        
    p_inf : float 
        Value of a certain property of the system at the limit of time
        approaching infinity. In other words, the value of the property for a
        fully relaxed system.

    Returns
    -------
    property_time : float or array_like
        Value of a certain property at different times during the relaxation
        process.

    References
    ----------
    [1] Kohlrausch, R.H.A. (1854). Theorie des elektrischen Rückstandes in der
        Leidener Flasche. Annalen Der Physik Und Chemie 167, 179–214.

    [2] Ngai, K.L. (2011). Relaxation and Diffusion in Complex Systems (New
        York: Springer).

    '''
    property_time = kohlrausch(time, tau_k, beta)*(p_0 - p_inf) + p_inf
    return property_time


