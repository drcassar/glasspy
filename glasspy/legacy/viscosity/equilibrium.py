"""Equations for equilibrium viscosity."""

from numpy import exp, log, log10
import numpy as np

from glasspy.viscosity.equilibrium_log import _belowT0correction


def clu(T, pre_exp, A, T0):
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
    viscosity = pre_exp * T ** (1 / 2) * (A / (T - T0))
    viscosity = _belowT0correction(T, T0, viscosity)
    return viscosity


def bs(T, eta_inf, A, T0, gamma=1):
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
    viscosity = eta_inf * exp(A / (T - T0) ** (3 * gamma / 2))
    viscosity = _belowT0correction(T, T0, viscosity)
    return viscosity


def dienes(T, eta_inf, A, B, T0):
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
    viscosity = _belowT0correction(T, T0, viscosity)
    return viscosity


def dml(T, eta_inf, A, B, T0):
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
    viscosity = _belowT0correction(T, T0, viscosity)
    return viscosity
