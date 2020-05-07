'''Equations for the base-10 logarithm of equilibrium viscosity.'''

from numpy import exp, log


def MYEGA(T, log_eta_inf, T12, m):
    """
    Computes the base-10 log of viscosity using the MYEGA equation.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_eta_inf : float
        Base-10 logarithm of the asymptotic viscosity at the limit of infinite
        temperature.

    T12 : float
        Temperature were the viscosity is 10**12 Pa.s. Unit: Kelvin.

    m : float
        Fragility index as defined by Angell, see refs. [2]. Unitless.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    [2] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
        Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
        Research Laboratory), pp. 3–12.

    """
    log10_viscosity = log_eta_inf + \
        (12-log_eta_inf)*(T12/T)*exp((m/(12-log_eta_inf)-1)*(T12/T-1))

    return log10_viscosity


def MYEGA_alt(T, log_eta_inf, K, C):
    """
    Computes the viscosity using the equation developed by Mauro and co-authors

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_eta_inf : float
        Base-10 logarithm of the asymptotic viscosity at the limit of infinite
        temperature.

    K : float
        See the original reference for the meaning. Unit: Kelvin.

    C : float
        See the original reference for the meaning. Unit: Kelvin.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    Notes
    -----
    In the original reference the equation is in base-10 logarithm, see Eq. (6)
    in [1]. To maintain the same meaning for the parameter K a log(10) is
    included in the double exponential expression here.

    References
    ----------
    [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    """
    log10_viscosity = log_eta_inf + K / T * exp(C / T)

    return log10_viscosity


def VFT(T, log_eta_inf, A, T0):
    """
    Computes the base-10 log of viscosity using the Vogel-Fulcher-Tammann eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_eta_inf : float
        Base-10 logarithm of the asymptotic viscosity at the limit of infinite
        temperature.

    A : float
        Adjustable parameter inside the exponential. Unit: Kelvin.

    T0 : float
        Divergence temperature. Unit: Kelvin.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Vogel, H. (1921). Das Temperatureabhängigketsgesetz der Viskosität von
        Flüssigkeiten. Physikalische Zeitschrift 22, 645–646.

    [2] Fulcher, G.S. (1925). Analysis of recent measurements of the viscosity
        of glasses. Journal of the American Ceramic Society 8, 339–355.

    [3] Tammann, G., and Hesse, W. (1926). Die Abhängigkeit der Viscosität von
        der Temperatur bie unterkühlten Flüssigkeiten. Z. Anorg. Allg. Chem.
        156, 245–257.

    """
    log10_viscosity = log_eta_inf + A / (T - T0)

    return log10_viscosity

