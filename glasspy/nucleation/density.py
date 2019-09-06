import numpy as np
from numpy import exp
from scipy.constants import pi
from scipy.special import exp1


def wakeshima(time, steady_state_rate, time_lag):
    """
    Computes the nuclei density using the Wakeshima equation.

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
    nuclei_density : float or array_like
        Returns the nuclei density.

    References
    ----------
    [1] Wakeshima, H. (1954). Time Lag in the Self‐Nucleation. The Journal of
        Chemical Physics 22, 1614–1615.
    """
    time_ratio = time/time_lag
    nuclei_density = steady_state_rate*(time_lag*(exp(-time_ratio) - 1) + time)
    return nuclei_density


def kashchiev(time, steady_state_rate, time_lag, time_shift=0,
              summation_ub=1000):
    """
    Computes the nuclei density using the Kashchiev equation.

    Parameters
    ----------
    time : float or array_like
        Elapsed time.

    steady_state_rate : float
        Steady-state nucleation rate.

    time_lag : float
        Time-lag of transient nucleation. See ref. [1].

    time_shift : float, optional
        Time-shift to account for double-stage treatments. See Eq. (33.65) in
        ref. [2]. Default value is 0.

    summation_ub : int, optional
        Upper boundary of the infinite summation. It is advisable to choose an
        even integer. Default value is 1000.

    Returns
    -------
    nuclei_density : float or array_like
        Returns the nuclei density.

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
    if time_lag == 0:
        nuclei_density = steady_state_rate*time

    else:

        @np.vectorize
        def _kashchiev(t):
            if t <= time_shift:
                return 0

            time_ratio = (t - time_shift)/time_lag

            def summationParticle(n):
                return ((-1)**(n%2))*exp(-n**2*time_ratio)/n**2

            summation = np.sum(summationParticle(np.arange(1, summation_ub + 1)))
            N = steady_state_rate*time_lag*(time_ratio - pi**2/6 - 2*summation)
            return N if N > 0 else 0

        nuclei_density = _kashchiev(time)

    return nuclei_density


def kashchievMasterCurve(time_ratio, summation_ub=1000):
    """
    Computes the normalized nuclei density using the Kashchiev equation.

    Parameters
    ----------
    time_ratio : float or array_like
        The ratio between the difference of time and time-shift over time-lag.

    summation_ub : int, optional
        Upper boundary of the infinite summation. It is advisable to choose an
        even integer. Default value is 1000.

    Returns
    -------
    normalized_nuclei_density : float or array_like
        Returns the normalized nuclei density, which is the nuclei density over
        the product of the steady-state rate and the time-lag.

    References
    ----------
    [1] Kashchiev, D. (1969). Solution of the non-steady state problem in
        nucleation kinetics. Surface Science 14, 209–220.

    [2] Kashchiev, D. (2000). Nucleation basic theory with applications.
    """
    @np.vectorize
    def _kashchiev(time_ratio):

        def summationParticle(n):
            return ((-1)**(n%2))*exp(-n**2*time_ratio)/n**2

        summation = np.sum(summationParticle(np.arange(1, summation_ub + 1)))
        Normalized_N = time_ratio - pi**2/6 - 2*summation
        return Normalized_N if Normalized_N > 0 else 0

    normalized_nuclei_density = _kashchiev(time_ratio)
    return normalized_nuclei_density


def shneidman(time, steady_state_rate, time_lag, incubation_time):
    """
    Computes the nuclei density using the Shneidman equation.

    Parameters
    ----------
    time : float or array_like
        Elapsed time.

    steady_state_rate : float
        Steady-state nucleation rate.

    time_lag : float
        Time-lag of transient nucleation. See Ref. [1].

    incubation_time : float
        Incubation time due to the double-stage treatment. See Ref. [1].

    Returns
    -------
    nuclei_density : float or array_like
        Returns the nuclei density.

    References
    ----------
    [1] Shneidman, V.A. (1988). Establishment of a steady-state nucleation
        regime. Theory and comparison with experimental data for glasses. Sov.
        Phys. Tech. Phys. 33, 1338–1342.
    """

    if time_lag == 0:
        nuclei_density = steady_state_rate*time

    else:

        time_ratio = (time - incubation_time)/time_lag
        nuclei_density = time_lag*steady_state_rate*exp1(exp(-time_ratio))

    return nuclei_density
