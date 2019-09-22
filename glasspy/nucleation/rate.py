import numpy as np
from numpy import exp


def wakeshima(time, steady_state_rate, time_lag):
    """
    Computes the time-dependent nucleation rate using the Wakeshima equation.

    Parameters
    ----------
    time : float or array_like
        Elapsed time.

    steady_state_rate : float
        Steady-state nucleation rate.

    time_lag : float
        Time-lag of transient nucleation. See ref. [1].

    Returns
    -------
    nucleation_rate : float or array_like
        Returns the nucleation rate.

    References
    ----------
    [1] Wakeshima, H. (1954). Time Lag in the Self‐Nucleation. The Journal of
        Chemical Physics 22, 1614–1615.

    [2] Wakeshima, H. (1954). Fog Formation due to Self Nucleation. J. Phys.
        Soc. Jpn. 9, 400–406.

    [3] Wakeshima, H. (1955). Development of Emulsions due to Self Nucleation.
        Journal of the Physical Society of Japan 10, 65–70.

    [4] Wakeshima, H. (1955). Time Lag in the Self-Nucleation. J. Phys. Soc.
        Jpn. 10, 374–380.

    [5] Wakeshima, H. (1955). Errata: Time Lag in the Self-Nucleation. The
        Journal of Chemical Physics 23, 763–763.

    """
    time_ratio = time / time_lag
    nucleation_rate = steady_state_rate * (1 - exp(-time_ratio))
    return nucleation_rate


def kashchiev(time,
              steady_state_rate,
              time_lag,
              time_shift=0,
              summation_ub=1000):
    """
    Computes the time-dependent nucleation rate using the Kashchiev equation.

    Parameters
    ----------
    time : float or array_like
        Elapsed time.

    steady_state_rate : float
        Steady-state nucleation rate.

    time_lag : float
        Time-lag of transient nucleation. See ref. [1].

    time_shift : float, optional
        Time-shift to account for double-stage treatments. See Eq. (33.66) in
        ref. [2]. Default value is 0.

    summation_ub : int, optional
        Upper boundary of the infinite summation. Default value is 1000. It is
        advisable to choose an even integer.

    Returns
    -------
    out : float or array_like
        Returns the nucleation rate.

    Notes
    -----
    This is the expression with the time-shift factor. For the original
    expression deduced in Ref. [1], set time_shift to zero.

    References
    ----------
    [1] Kashchiev, D. (1969). Solution of the non-steady state problem in
        nucleation kinetics. Surface Science 14, 209–220.

    [2] Kashchiev, D. (2000). Nucleation basic theory with applications.
    """
    @np.vectorize
    def _kashchiev(t):
        if t <= time_shift:
            return 0

        try:
            time_ratio = (t - time_shift) / time_lag
        except ZeroDivisionError:
            time_ratio = np.inf

        def summationParticle(n):
            return ((-1)**(n % 2)) * exp(-n**2 * time_ratio)

        summation = np.sum(summationParticle(np.arange(1, summation_ub)))
        I = steady_state_rate * (1 + 2 * summation)
        return I if I > 0 else 0

    nucleation_rate = _kashchiev(time)
    return nucleation_rate


def kashchievMasterCurve(time_ratio, summation_ub=1000):
    """
    Computes the normalized nucleation rate using the Kashchiev equation.

    Parameters
    ----------
    time_ratio : float or array_like
        The ratio between the difference of time and time-shift over time-lag.

    summation_ub : int, optional
        Upper boundary of the infinite summation. Default value is 1000. It is
        advisable to choose an even integer.

    Returns
    -------
    normalized_nucleation_rate : float or array_like
        Returns the ration between the nucleation rate and the steady-state
        nucleation rate.

    References
    ----------
    [1] Kashchiev, D. (1969). Solution of the non-steady state problem in
        nucleation kinetics. Surface Science 14, 209–220.

    [2] Kashchiev, D. (2000). Nucleation basic theory with applications.
    """
    @np.vectorize
    def _kashchiev(time_ratio):

        if time_ratio < 0.1:
            return 0.
        elif time_ratio > 20:
            return 1.

        def summationParticle(n):
            return ((-1)**(n % 2)) * exp(-n**2 * time_ratio)

        summation = np.sum(summationParticle(np.arange(1, summation_ub)))
        normalized_I = 1 + 2 * summation
        return normalized_I if normalized_I > 0 else 0

    normalized_nucleation_rate = _kashchiev(time_ratio)
    return normalized_nucleation_rate


def shneidman(time, steady_state_rate, time_lag, time_incubation):
    """
    Computes the time-dependent nucleation rate using the Shneidman equation.

    Parameters
    ----------
    time : float or array_like
        Elapsed time.

    steady_state_rate : float
        Steady-state nucleation rate.

    time_lag : float
        Time-lag of transient nucleation. See Ref. [1].

    time_incubation : float
        Incubation time due to the double-stage treatment. See Ref. [1].

    Returns
    -------
    out : float or array_like
        Returns the nucleation rate.

    References
    ----------
    [1] Shneidman, V.A. (1988). Establishment of a steady-state nucleation
        regime. Theory and comparison with experimental data for glasses. Sov.
        Phys. Tech. Phys. 33, 1338–1342.
    """
    time_ratio = (time - time_incubation) / time_lag
    nucleation_rate = steady_state_rate * exp(-exp(-time_ratio))
    return nucleation_rate
