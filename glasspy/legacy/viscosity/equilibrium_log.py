"""Equations for the base-10 logarithm of equilibrium viscosity."""

from numpy import exp, log, log10
import numpy as np


def _belowT0correction(T, T0, viscosity):
    """Returns infinity viscosity below the divergence temperature (T0)."""
    return np.where(np.less_equal(T, T0), np.inf, viscosity)


def clu(T, log_pre_exp, A, T0):
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
    log10_viscosity = _belowT0correction(T, T0, log10_viscosity)
    return log10_viscosity


def bs(T, log_eta_inf, A, T0, gamma=1):
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
    log10_viscosity = log_eta_inf + A / (T - T0) ** (3 * gamma / 2) / log(10)
    log10_viscosity = _belowT0correction(T, T0, log10_viscosity)
    return log10_viscosity


def dienes(T, log_eta_inf, A, B, T0):
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
    log10_viscosity = (
        log_eta_inf
        - log10(2)
        + (B / T * log(10))
        + log10(exp(A / (T - T0)) + 1)
    )
    log10_viscosity = _belowT0correction(T, T0, log10_viscosity)
    return log10_viscosity


def dml(T, log_eta_inf, A, B, T0):
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
    log10_viscosity = (
        log_eta_inf + B / (T * log(10)) + A / ((T - T0) * log(10))
    )
    log10_viscosity = _belowT0correction(T, T0, log10_viscosity)
    return log10_viscosity
