'''Equations for equilibrium viscosity.'''

from numpy import exp, log, log10


def MYEGA(T, eta_inf, K, C):
    """
    Computes the viscosity using the equation developed by Mauro and co-authors

    Mathematicaly, this equation is the same as that proposed in ref. [2] (see
    page 250), however the physical considerations are different.

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
    viscosity = eta_inf * 10**(K / T * exp(C / T))

    return viscosity


def MYEGA_alt(T, eta_inf, T12, m):
    """
    Computes the viscosity using the equation developed by Mauro and co-authors

    This is an alternate form of the MYEGA equation found in [1]

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

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
    log_eta_inf = log10(eta_inf)
    viscosity = eta_inf * 10**(
        (12-log_eta_inf)*(T12/T)*exp((m/(12-log_eta_inf)-1)*(T12/T-1))
    )

    return viscosity


def VFT(T, eta_inf, A, T0):
    """
    Computes the viscosity using the empirical Vogel-Fulcher-Tammann eq.

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


def VFT_alt(T, eta_inf, T12, m):
    """
    Computes the viscosity using the Vogel-Fulcher-Tammann eq.

    This is the rewriten VFT equation found in ref. [4].
    
    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    T12 : float
        Temperature were the viscosity is 10**12 Pa.s. Unit: Kelvin.

    m : float
        Fragility index as defined by Angell, see refs. [5]. Unitless.

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

    [4] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    [5] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
        Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
        Research Laboratory), pp. 3–12.

    """
    log_eta_inf = log10(eta_inf)
    viscosity = log_eta_inf * 10**((12 - log_eta_inf)**2 / \
        (m * (T / T12 - 1) + (12 - log_eta_inf)))

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


def AM_alt(T, eta_inf, T12, m):
    """
    Computes the viscosity using the equation developed by Avramov & Milchev.

    This is the rewriten AM equation found in ref. [3].
    
    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    T12 : float
        Temperature were the viscosity is 10**12 Pa.s. Unit: Kelvin.

    m : float
        Fragility index as defined by Angell, see refs. [4]. Unitless.

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

    [3] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
        (2009). Viscosity of glass-forming liquids. Proceedings of the National
        Academy of Sciences of the United States of America 106, 19780–19784.

    [4] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
        Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
        Research Laboratory), pp. 3–12.

    """
    log_eta_inf = log10(eta_inf)
    viscosity = log_eta_inf * 10**((12 - log_eta_inf) * \
        (T12 / T)**(m/(12 - log_eta_inf)))

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
    viscosity = eta_inf * exp(B / (T * S_conf_fun(T)))

    return viscosity


def CLU(T, pre_exp, A, T0):
    """
    Computes the viscosity using the Cukierman-Lane-Uhlmann eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    pre_exp : float
        Pre-exponential factor.

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
    [1] Cukierman, M., Lane, J.W., and Uhlmann, D.R. (1973). High‐temperature
        flow behavior of glass‐forming liquids: A free‐volume interpretation.
        The Journal of Chemical Physics 59, 3639–3644.

    """
    viscosity = pre_exp*T**(1/2)*(A / (T - T0))

    return viscosity


def BS(T, eta_inf, A, T0, gamma=1):
    """
    Computes the viscosity using the Bendler-Shlesinger eq.

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

    gamma : float
        See ref. [1].

    Returns
    -------
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

    References
    ----------
    [1] Bendler, J.T., and Shlesinger, M.F. (1988). Generalized Vogel law for
        glass-forming liquids. J Stat Phys 53, 531–541.

    """
    viscosity = eta_inf * exp(A / (T - T0)**(3 * gamma / 2))

    return viscosity


def Dienes(T, eta_inf, A, B, T0):
    """
    Computes the viscosity using the Dienes eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    A : float
        Adjustable parameter. Unit: Kelvin.

    B : float
        Adjustable parameter. Unit: Kelvin.

    T0 : float
        Divergence temperature. Unit: Kelvin.

    Returns
    -------
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

    References
    ----------
    [1] Dienes, G.J. (1953). Activation Energy for Viscous Flow and Short‐Range
        Order. Journal of Applied Physics 24, 779–782.

    """
    viscosity = eta_inf / 2 * exp(B / T) * (exp(A / (T - T0)) + 1)

    return viscosity


def DML(T, eta_inf, A, B, T0):
    """
    Computes the viscosity using the DML eq.

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    eta_inf : float
        Asymptotic viscosity at the limit of infinite temperature.

    A : float
        Adjustable parameter. Unit: Kelvin.

    B : float
        Adjustable parameter. Unit: Kelvin.

    T0 : float
        Divergence temperature. Unit: Kelvin.

    Returns
    -------
    viscosity : float or array_like
        Returns the viscosity in the units of eta_inf. Note: it is *not* the
        logarithm of viscosity.

    References
    ----------
    [1] Dienes, G.J. (1953). Activation Energy for Viscous Flow and Short‐Range
        Order. Journal of Applied Physics 24, 779–782.

    [2] Macedo, P.B., and Litovitz, T.A. (1965). On the Relative Roles of Free
        Volume and Activation Energy in the Viscosity of Liquids. The Journal of
        Chemical Physics 42, 245–256.

    """
    viscosity = eta_inf * exp(A / (T - T0) + B / T)

    return viscosity
