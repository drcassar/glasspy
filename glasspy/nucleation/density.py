import numpy as np
from numpy import exp
from scipy.constants import pi


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
    out : float or array_like
        Returns the nuclei density.

    References
    ----------
    [1] Wakeshima, H. (1954). Time Lag in the Self‐Nucleation. The Journal of
        Chemical Physics 22, 1614–1615.
    """
    return steady_state_rate*(time_lag*(exp(-time/time_lag) - 1) + time)


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
        Upper boundary of the infinite summation. Default value is 1000. It is
        advisable to choose an even integer.

    Returns
    -------
    out : float or array_like
        Returns the nuclei density.

    Notes
    -----
    This is the expression with the time-shift factor. For the original
    expression deduced in Ref. [1], set time_shift to zero.

    References
    ----------
    [1] Kashchiev, D. (1969). Solution of the non-steady state problem in
        nucleation kinetics. Surface Science 14, 209–220.

    [2] Kashchiev, D. (2000). Nucleation basic theory with applications
        (Oxford; Boston: Butterworth Heinemann).
    """
    @np.vectorize
    def _kashchiev(t):
        if t <= time_shift:
            return 0

        try:
            time_ratio = (t - time_shift)/time_lag
        except ZeroDivisionError:
            time_ratio = np.inf

        def summationParticle(n):
            return ((-1)**(n%2))*exp(-n**2*tovertau)/n**2

        summation = np.sum(summationParticle(np.arrange(1, summation_ub)))
        N = steady_state_rate*time_lag*(time_ratio - pi**2/6 - 2*summation)
        return N if N > 0 else 0

    nuclei_density = _kashchiev(time)
    return nuclei_density
