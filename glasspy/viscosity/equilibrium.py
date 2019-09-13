'''Equations for equilibrium viscosity.'''

from numpy import exp, log


def logMYEGA(T, log_eta_inf, T12, m):
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
        Returns the base-10 logarithmo of viscosity.

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


def MYEGA(T, log_eta_inf, T12, m):
    """
    Computes the viscosity using the equation developed by Mauro and co-authors

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
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

    References
    ----------
    [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    [2] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
        Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
        Research Laboratory), pp. 3–12.

    """
    log10_viscosity = logMYEGA(T, log_eta_inf, T12, m)
    viscosity = 10**log10_viscosity

    return viscosity


def MYEGA_alt(T, eta_inf, K, C):
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
    included in the double exponential expression here.

    References
    ----------
    [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.
    """
    viscosity = eta_inf * exp(log(10) * K / T * exp(C / T))

    return viscosity


def VFT(T, eta_inf, A, T0):
    """
    Computes the viscosity using the empirical equation Vogel-Fulcher-Tammann eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    A : float
        Adjustable parameter inside the exponential. Unit: Kelvin.

    T0 : float
        Divergence temperature. Unit: Kelvin.

    Returns
    -------
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

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
    viscosity = eta_inf * exp(log(10) * A / (T - T0))
    return viscosity


def AM(T, eta_inf, alpha, beta):
    """
    Computes the viscosity using the equation developed by Avramov & Milchev.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    alpha : float
        Adjustable parameter, see original reference. Unitless.

    beta : float
        Adjustable parameter with unit of Kelvin.

    Returns
    -------
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

    References
    ----------
    [1] Avramov, I., and Milchev, A. (1988). Effect of disorder on diffusion
        and viscosity in condensed systems. Journal of Non-Crystalline Solids
        104, 253–260.

    [2] Cornelissen, J., and Waterman, H.I. (1955). The viscosity temperature
        relationship of liquids. Chemical Engineering Science 4, 238–246.
    """
    viscosity = eta_inf * exp((beta / T)**alpha)
    return viscosity


def AG(T, eta_inf, B, S_conf_fun):
    """
    Computes the viscosity using the equation developed by Adam & Gibbs

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    B : float
        Adjustable parameter related to the potential energy hindering the
        cooperative rearrangement per monomer segment.

    S_conf_fun : callable
        Function that computes the configurational entropy. This function
        accepts one argument, that is temperature.

    Returns
    -------
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

    References
    ----------
    [1] Adam, G., and Gibbs, J.H. (1965). On the temperature dependence of
        cooperative relaxation properties in glass-forming liquids. The Journal
        of Chemical Physics 43, 139–146.
    """
    viscosity = eta_inf * exp(-B / (T * S_conf_fun(T)))

    return viscosity
