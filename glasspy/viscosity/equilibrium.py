'''Equations for equilibrium viscosity.'''

from numpy import exp, log


def MYEGA(T, eta_inf, K, C):
    """
    Computes the viscosity using the equation developed by Mauro and co-authors

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    K : float
        See the original reference for the meaning. Unit: Kelvin.

    C : float
        See the original reference for the meaning. Unit: Kelvin.

    Returns
    -------
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

    Notes
    -----
    In the original reference the equation is in base-10 logarithm, see Eq. (6)
    in [1]. To maintain the same meaning for the parameter K a log(10) is
    included in the expression.

    References
    ----------
    [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.
    """

    viscosity = eta_inf * exp(log(10) * K / T * exp(C / T))
    return viscosity


