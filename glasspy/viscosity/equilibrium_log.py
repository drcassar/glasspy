"""Equations for the base-10 logarithm of equilibrium viscosity."""

from numpy import exp, log, log10, where, inf, less_equal


def _belowT0correction(T, T0, viscosity):
    """Returns infinity viscosity below the divergence temperature (T0)."""
    return where(less_equal(T, T0), inf, viscosity)


def myega(T, log_eta_inf, K, C):
    """Computes the viscosity using the MYEGA equation.

    Mathematicaly, this equation is the same as that proposed in ref. [2] (see
    page 250), however the physical considerations are different.

    Args:
      T : float or array_like
          Temperature. Unit: Kelvin.
      eta_inf : float
          Asymptotic viscosity at the limit of infinite temperature.
      K : float
          See the original reference for the meaning. Unit: Kelvin.
      C : float
          See the original reference for the meaning. Unit: Kelvin.

    Returns:
      Returns the base-10 logarithm of viscosity.

    Notes:
      In the original reference the equation is in base-10 logarithm, see Eq.
      (6) in [1].

    References
    ----------
      [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
          (2009). Viscosity of glass-forming liquids. Proceedings of the
          National Academy of Sciences of the United States of America 106,
          19780–19784.

      [2] Waterton, S.C. (1932). The viscosity-temperature relationship and some
          inferences on the nature of molten and of plastic glass. J Soc Glass
          Technol 16, 244–249.
    """

    log10_viscosity = log_eta_inf + K / T * exp(C / T)

    return log10_viscosity


def myega_alt(T, log_eta_inf, T12, m):
    """ Computes the viscosity using the MYEGA equation.

    This is an alternate form of the MYEGA equation found in [1]

    Args:
      T : float or array_like
          Temperature. Unit: Kelvin.
      eta_inf : float
          Asymptotic viscosity at the limit of infinite temperature.
      T12 : float
          Temperature were the viscosity is 10**12 Pa.s. Unit: Kelvin.
      m : float
          Fragility index as defined by Angell, see ref. [2]. Unitless.

    Returns:
      Returns the base-10 logarithm of viscosity.

    References:
      [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
          (2009). Viscosity of glass-forming liquids. Proceedings of the
          National Academy of Sciences of the United States of America 106,
          19780–19784.

      [2] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
          Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
          Research Laboratory), pp. 3–12.
    """

    log10_viscosity = log_eta_inf + (12 - log_eta_inf) * (T12 / T) * exp(
        (m / (12 - log_eta_inf) - 1) * (T12 / T - 1)
    )

    return log10_viscosity


def vft(T, log_eta_inf, A, T0):
    """Computes the viscosity using the empirical Vogel-Fulcher-Tammann eq.

    Args:
      T : float or array_like
          Temperature. Unit: Kelvin.
      eta_inf : float
          Asymptotic viscosity at the limit of infinite temperature.
      A : float
          Adjustable parameter inside the exponential. Unit: Kelvin.
      T0 : float
          Divergence temperature. Unit: Kelvin.

    Returns:
      Returns the base-10 logarithm of viscosity.

    References:
      [1] Vogel, H. (1921). Das Temperatureabhängigketsgesetz der Viskosität von
          Flüssigkeiten. Physikalische Zeitschrift 22, 645–646.

      [2] Fulcher, G.S. (1925). Analysis of recent measurements of the viscosity
          of glasses. Journal of the American Ceramic Society 8, 339–355.

      [3] Tammann, G., and Hesse, W. (1926). Die Abhängigkeit der Viscosität von
          der Temperatur bie unterkühlten Flüssigkeiten. Z. Anorg. Allg. Chem.
          156, 245–257.
    """

    log10_viscosity = log_eta_inf + A / (T - T0)
    log10_viscosity = _belowT0correction(T, T0, log10_viscosity)
    return log10_viscosity


def vft_alt(T, log_eta_inf, T12, m):
    """Computes the viscosity using the Vogel-Fulcher-Tammann eq.

    This is the rewriten VFT equation found in ref. [4].

    Args:
      T : float or array_like
          Temperature. Unit: Kelvin.
      eta_inf : float
          Asymptotic viscosity at the limit of infinite temperature.
      T12 : float
          Temperature were the viscosity is 10**12 Pa.s. Unit: Kelvin.
      m : float
          Fragility index as defined by Angell, see ref. [5]. Unitless.

    Returns:
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
          (2009). Viscosity of glass-forming liquids. Proceedings of the
          National Academy of Sciences of the United States of America 106,
          19780–19784.

      [5] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
          Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
          Research Laboratory), pp. 3–12.
    """

    log10_viscosity = log_eta_inf + (12 - log_eta_inf) ** 2 / (
        m * (T / T12 - 1) + (12 - log_eta_inf)
    )
    T0 = T12 * (1 - (12 - log_eta_inf) / m)
    log10_viscosity = _belowT0correction(T, T0, log10_viscosity)
    return log10_viscosity


def am(T, log_eta_inf, alpha, beta):
    """Computes the viscosity using the Avramov & Milchev equation.

    Args:
      T : float or array_like
          Temperature. Unit: Kelvin.
      eta_inf : float
          Asymptotic viscosity at the limit of infinite temperature.
      alpha : float
          Adjustable parameter, see original reference. Unitless.
      beta : float
          Adjustable parameter with unit of Kelvin.

    Returns:
      Returns the base-10 logarithm of viscosity.

    References:
      [1] Avramov, I., and Milchev, A. (1988). Effect of disorder on diffusion
          and viscosity in condensed systems. Journal of Non-Crystalline Solids
          104, 253–260.
      [2] Cornelissen, J., and Waterman, H.I. (1955). The viscosity temperature
          relationship of liquids. Chemical Engineering Science 4, 238–246.
    """

    log10_viscosity = log_eta_inf + (beta / T) ** alpha / log(10)

    return log10_viscosity


def am_alt(T, log_eta_inf, T12, m):
    """Computes the viscosity using the Avramov & Milchev equation.

    This is the rewriten AM equation found in ref. [3].

    Args:
      T : float or array_like
          Temperature. Unit: Kelvin.
      eta_inf : float
          Asymptotic viscosity at the limit of infinite temperature.
      T12 : float
          Temperature were the viscosity is 10**12 Pa.s. Unit: Kelvin.
      m : float
          Fragility index as defined by Angell, see ref. [4]. Unitless.

    Returns:
      Returns the base-10 logarithm of viscosity.

    References:
      [1] Avramov, I., and Milchev, A. (1988). Effect of disorder on diffusion
          and viscosity in condensed systems. Journal of Non-Crystalline Solids
          104, 253–260.
      [2] Cornelissen, J., and Waterman, H.I. (1955). The viscosity temperature
          relationship of liquids. Chemical Engineering Science 4, 238–246.
      [3] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
          (2009). Viscosity of glass-forming liquids. Proceedings of the
          National Academy of Sciences of the United States of America 106,
          19780–19784.
      [4] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
          Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield: Naval
          Research Laboratory), pp. 3–12.
    """

    log10_viscosity = log_eta_inf + (12 - log_eta_inf) * (T12 / T) ** (
        m / (12 - log_eta_inf)
    )

    return log10_viscosity


def ag(T, eta_inf, B, S_conf_fun):
    """Computes the viscosity using the Adam & Gibbs equation.

    Args:
      T : float or array_like
          Temperature. Unit: Kelvin.
      eta_inf : float
          Asymptotic viscosity at the limit of infinite temperature.
      B : float
          Adjustable parameter related to the potential energy hindering the
          cooperative rearrangement per monomer segment.
      S_conf_fun : callable
          Function that computes the configurational entropy. This function
          accepts one argument, which is the absolute temperature.

    Returns:
      Returns the base-10 logarithm of viscosity.

    References:
      [1] Adam, G., and Gibbs, J.H. (1965). On the temperature dependence of
          cooperative relaxation properties in glass-forming liquids. The
          Journal of Chemical Physics 43, 139–146.
    """

    log10_viscosity = log_eta_inf + B / (T * S_conf_fun(T) * log(10))

    return log10_viscosity

