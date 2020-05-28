'''Equations for the base-10 logarithm of equilibrium viscosity.'''

from numpy import exp, log, log10


def MYEGA(T, log_eta_inf, K, C):
    """
    Computes the viscosity using the equation developed by Mauro and co-authors

    Mathematicaly, this equation is the same as that proposed in ref. [2] (see
    page 250), however the physical considerations are different.

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
    in [1]. 

    References
    ----------
    [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    [2] Waterton, S.C. (1932). The viscosity-temperature relationship and some
        inferences on the nature of molten and of plastic glass. J Soc Glass
        Technol 16, 244–249.

    """
    log10_viscosity = log_eta_inf + K / T * exp(C / T)

    return log10_viscosity


def MYEGA_alt(T, log_eta_inf, T12, m):
    """
    Computes the base-10 log of viscosity using the MYEGA equation.

    This is an alternate form of the MYEGA equation found in [1]

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


def VFT_alt(T, log_eta_inf, T12, m):
    """
    Computes the base-10 log of viscosity using the Vogel-Fulcher-Tammann eq.

    This is the rewriten VFT equation found in ref. [4].
    
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
        Fragility index as defined by Angell, see refs. [5]. Unitless.

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

    [4] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    [5] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
        Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
        Research Laboratory), pp. 3–12.

    """
    log10_viscosity = log_eta_inf + (12 - log_eta_inf)**2 / \
        (m * (T / T12 - 1) + (12 - log_eta_inf))

    return log10_viscosity


def AM(T, log_eta_inf, alpha, beta):
    """
    Computes the base-10 log of viscosity using the Avramov & Milchev eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_eta_inf : float
        Base-10 logarithm of the asymptotic viscosity at the limit of infinite
        temperature.

    alpha : float
        Adjustable parameter, see original reference. Unitless.

    beta : float
        Adjustable parameter with unit of Kelvin.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Avramov, I., and Milchev, A. (1988). Effect of disorder on diffusion
        and viscosity in condensed systems. Journal of Non-Crystalline Solids
        104, 253–260.

    [2] Cornelissen, J., and Waterman, H.I. (1955). The viscosity temperature
        relationship of liquids. Chemical Engineering Science 4, 238–246.

    """
    log10_viscosity = log_eta_inf + (beta / T)**alpha / log(10)

    return log10_viscosity


def AM_alt(T, log_eta_inf, T12, m):
    """
    Computes the base-10 log of viscosity using the Avramov & Milchev eq.

    This is the rewriten AM equation found in ref. [3].
    
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
        Fragility index as defined by Angell, see refs. [4]. Unitless.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Avramov, I., and Milchev, A. (1988). Effect of disorder on diffusion
        and viscosity in condensed systems. Journal of Non-Crystalline Solids
        104, 253–260.

    [2] Cornelissen, J., and Waterman, H.I. (1955). The viscosity temperature
        relationship of liquids. Chemical Engineering Science 4, 238–246.

    [3] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    [4] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
        Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
        Research Laboratory), pp. 3–12.

    """
    log10_viscosity = log_eta_inf + (12 - log_eta_inf) * \
        (T12 / T)**(m/(12 - log_eta_inf))

    return log10_viscosity


def AG(T, eta_inf, B, S_conf_fun):
    """
    Computes the base-10 log of viscosity using the Adam & Gibbs eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_eta_inf : float
        Base-10 logarithm of the asymptotic viscosity at the limit of infinite
        temperature.

    B : float
        Adjustable parameter related to the potential energy hindering the
        cooperative rearrangement per monomer segment.

    S_conf_fun : callable
        Function that computes the configurational entropy. This function
        accepts one argument, that is temperature.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Adam, G., and Gibbs, J.H. (1965). On the temperature dependence of
        cooperative relaxation properties in glass-forming liquids. The Journal
        of Chemical Physics 43, 139–146.

    """
    log10_viscosity = log_eta_inf + B / (T * S_conf_fun(T) * log(10))

    return log10_viscosity


def CLU(T, log_pre_exp, A, T0):
    """
    Computes the base-10 log of viscosity using the Cukierman-Lane-Uhlmann eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_pre_exp : float
        Base-10 logarithm of the pre-exponential factor.

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
    [1] Cukierman, M., Lane, J.W., and Uhlmann, D.R. (1973). High‐temperature
        flow behavior of glass‐forming liquids: A free‐volume interpretation.
        The Journal of Chemical Physics 59, 3639–3644.

    """
    log10_viscosity = log_pre_exp + log10(T) / 2 + A / (T - T0)

    return log10_viscosity


def BS(T, log_eta_inf, A, T0, gamma=1):
    """
    Computes the base-10 log of viscosity using the BS eq.

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

    gamma : float
        See ref. [1].

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Bendler, J.T., and Shlesinger, M.F. (1988). Generalized Vogel law for
        glass-forming liquids. J Stat Phys 53, 531–541.

    """
    log10_viscosity = log_eta_inf + A / (T - T0)**(3 * gamma / 2) / log(10)

    return log10_viscosity


def Dienes(T, log_eta_inf, A, B, T0):
    """
    Computes the base-10 log of viscosity using the Dienes eq.

    This is eq. (9) in ref. [1]

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_eta_inf : float
        Base-10 logarithm of the asymptotic viscosity at the limit of infinite
        temperature.

    A : float
        Adjustable parameter inside the exponential. Unit: Kelvin.

    B : float
        Adjustable parameter. Unit: Kelvin.

    T0 : float
        Divergence temperature. Unit: Kelvin.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Dienes, G.J. (1953). Activation Energy for Viscous Flow and Short‐Range
        Order. Journal of Applied Physics 24, 779–782.

    """
    log10_viscosity = log_eta_inf - log10(2) + (B / T * log(10)) + \
        log10(exp(A / (T - T0)) + 1)

    return log10_viscosity


def DML(T, log_eta_inf, A, B, T0):
    """
    Computes the base-10 log of viscosity using the DML eq.

    This is eq. (10) in ref. [1] and eq. (19) in ref. [2]

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    log_eta_inf : float
        Base-10 logarithm of the asymptotic viscosity at the limit of infinite
        temperature.

    A : float
        Adjustable parameter inside the exponential. Unit: Kelvin.

    B : float
        Adjustable parameter. Unit: Kelvin.

    T0 : float
        Divergence temperature. Unit: Kelvin.

    Returns
    -------
    log10_viscosity : float or array_like
        Returns the base-10 logarithm of viscosity.

    References
    ----------
    [1] Dienes, G.J. (1953). Activation Energy for Viscous Flow and Short‐Range
        Order. Journal of Applied Physics 24, 779–782.

    [2] Macedo, P.B., and Litovitz, T.A. (1965). On the Relative Roles of Free
        Volume and Activation Energy in the Viscosity of Liquids. The Journal of
        Chemical Physics 42, 245–256.

    """
    log10_viscosity = log_eta_inf + B / (T * log(10)) + A / ((T - T0) * log(10))

    return log10_viscosity
