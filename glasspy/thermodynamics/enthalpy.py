import numpy as np
from scipy.integrate import quad


def totalTransformationEnthalpy(H_of_transformations, lower_temperature,
                                upper_temperature):
    '''
    Computes the total transformation enthalpy between two temperatures

    Parameters
    ----------
    H_of_transformations : dictionary or None or False
        If 'None' or 'False' then the function returs zero, as it is understood
        that no phase transition happens (or is known) for this substance. If
        this argument is a dictionary then each key of the dictionary must be
        an integer or float representing the temperature were the transition
        occurs (temperature is expected to be in Kelvin) and the value of each
        key must be the enthalpy of transformation during heating (expected to
        be in J/mol), represented as an integer or float. See notes for an
        example.

    lower_temperature : integer or float
        Lower temperature range (expected to be in Kelvin) to be considered for
        the calculation. If 'lower_temperature' is higher than
        'upper_temperature' the function raises a ValueError.

    upper_temperature : integer or float
        Upper temperature range (expected to be in Kelvin) to be considered for
        the calculation. If 'lower_temperature' is higher than
        'upper_temperature' the function raises a ValueError.

    Returns
    -------
    total_transformation_enthalpy : float
        Sum of all the transformation enthalpy that happens between
        'lower_temperature' and 'upper_temperature'.

    Notes
    -----
    See below an example of the dictionary format for 'H_of_transformation' for
    Silica (SiO2, see page 1505 in Ref. [1]).

    H_of_transformations_SiO2 = {
        847: 728.0,  # from low quartz to high quartz (J/mol)
        1079: 1995.8,  # from high quartz to high cristobalite (J/mol)
    }

    References
    ----------
    [1] Barin, I. (1995). Thermochemical data of pure substances.

    '''
    total_transformation_enthalpy = 0

    if isinstance(H_of_transformations, dict):
        if lower_temperature > upper_temperature:
            raise \
                ValueError('lower_temperature is higher than upper_temperature')

        for temperature in H_of_transformations:
            if lower_temperature <= temperature <= upper_temperature:
                total_transformation_enthalpy += \
                    H_of_transformations[temperature]

    return total_transformation_enthalpy


def excessEnthalpyFun(Cp_crystal_fun,
                      Cp_liquid_fun,
                      Tm,
                      delta_Hm,
                      H_transformations=None):
    '''
    Generate a function to compute the excess enthalpy.

    The excess enthalpy is the enthalpy of the liquid phase minus the enthalpy
    of the crystalline phase.

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
        documentation for the function totalTransformationEnthalpy for more
        details. Default value is None.

    Returns
    -------
    HexcFun : callable
        Function with one argument for temperature (in Kelvin) that returns the
        excess enthalpy at the given temperature. The excess enthalpy is the
        difference between the liquid and the crystalline enthalpy at the given
        temperature The argument of the function can be a scalar or an
        array-like.

    '''
    def fun(x):
        return Cp_liquid_fun(x) - Cp_crystal_fun(x)

    @np.vectorize
    def HexcFun(T):

        sum_transformation_enthalpy = \
            totalTransformationEnthalpy(H_transformations, T, Tm)
        integral = quad(fun, Tm, T, limit=100)[0]
        excess_enthalpy = delta_Hm + integral + sum_transformation_enthalpy

        return excess_enthalpy

    return HexcFun
