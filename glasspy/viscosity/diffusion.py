'''Equations for computing the effective diffusion coefficient from
viscosity.'''

from scipy.constants import k, pi


def diffCoeffEyring(T, viscosity, diameter):
    """
    Computes the viscosity diffusion coefficient using Eyring equation

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    viscosity : float or array_like with same lenght as T
        Viscosity at temperature T.

    diameter : float or array_like with same lenght as T
        The diameter of the structural unit that is moving due to viscous flow.

    Returns
    -------
    out : float or array_like
        Returns the effective diffusion coefficient computed using the Eyring
        equation. This equation is similar to the Stokes-Einstein equation, but
        they were obtained by different routes.

    References
    ----------
    [1] Eyring, H. (1936). Viscosity, plasticity, and diffusion as examples of
        absolute reaction rates. The Journal of Chemical Physics 4, 283–291.
    """
    return k*T/(viscosity*diameter)


def diffCoeffStokesEinstein(T, viscosity, diameter):
    """
    Computes the viscosity diffusion coefficient using Stokes-Einstein equation

    Parameters
    ----------
    T : float or array_like
        Temperature. Unit: Kelvin.

    viscosity : float or array_like with same lenght as T
        Viscosity at temperature T.

    diameter : float or array_like with same lenght as T
        The diameter of the structural unit that is moving due to viscous flow.

    Returns
    -------
    out : float or array_like
        Returns the effective diffusion coefficient computed using the
        Stokes-Einstein equation. This equation is similar to the Eyring
        equation, but they were obtained by different routes.

    References
    ----------
    [1] Einstein, A. (1905). On the movement of small particles suspended in
        stationary liquids required by the molecular-kinetic theory of heat.
        Annalen Der Physik 17, 549–560.

    [2] Einstein, A. (1905). Über die von der molekularkinetischen Theorie der
        Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten
        Teilchen. Annalen Der Physik 322, 549–560.

    [3] Stokes, G.G. (1851). On the effect of the internal friction of fluids
        on the motion of pendulums. Transactions of the Cambridge Philosophical
        Society 9, 8–106.
    """
    return k*T/(3*pi*viscosity*diameter)
