import numpy as np
from scipy.integrate import quad
from .enthalpy import excessEnthalpyFun
from .entropy import excessEntropyFun


def excessGibbsFun(Cp_crystal_fun,
                   Cp_liquid_fun,
                   Tm,
                   delta_Hm,
                   H_transformations=None):
    '''
    Generate a function to compute the excess Gibbs free energy.

    The excess Gibbs free energy is the free energy of the liquid phase minus
    the free energy of the crystalline phase. The excess free energy, in the
    way it is defined here, must be positive below the melting point and
    negative above it. It is usual in the literature to call the excess Gibbs
    free energy as the crystallization driving force (not to be confused as the
    change in free energy due to crystallization, for which is just the
    negative of the excess Gibbs free energy).

    Parameters
    ----------
    Cp_crystal_fun : callable
        Function with one argument for the temperature (in Kelvin) that returns
        the heat capacity at constant pressure for the crystalline phase.

    Cp_liquid_fun : callable
        Function with one argument for the temperature (in Kelvin) that returns
        the heat capacity at constant pressure for the liquid phase.

    Tm : integer or float
        Melting temperature.

    delta_Hm : integer or float
        Enthalpy of melting.

    H_transformations : dictionary or None or False
        If there are any crystalline phase transitions in the material, this
        information must be given as a dictionary were the keys are the
        temperature where the transition happens (in Kelvin) and the values are
        the enthalpy of transformations (in J/mol) during heating. See the
        documentation for the function totalTransformationEntropy for more
        details. Default value is None.

    Returns
    -------
    GexcFun : callable
        Function with one argument for temperature (in Kelvin) that returns the
        excess Gibbs free energy at the given temperature. The excess Gibbs
        free energy is the difference between the liquid and the crystalline
        free energies at the given temperature The argument of the function can
        be a scalar or an array-like.

    '''
    HexcFun = excessEnthalpyFun(Cp_crystal_fun, Cp_liquid_fun, Tm, delta_Hm,
                                H_transformations)
    SexcFun = excessEntropyFun(Cp_crystal_fun, Cp_liquid_fun, Tm, delta_Hm,
                               H_transformations)

    def GexcFun(T):
        return HexcFun(T) - T * SexcFun(T)

    return GexcFun
