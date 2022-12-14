import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def totalTransformationEntropy(H_of_transformations, lower_temperature,
                               upper_temperature):
    '''
    Computes the total transformation entropy between two temperatures

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
    total_transformation_entropy : float
        Sum of all the transformation entropy that happens between
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
    total_transformation_entropy = 0

    if isinstance(H_of_transformations, dict):
        if lower_temperature > upper_temperature:
            raise \
                ValueError('lower_temperature is higher than upper_temperature')

        for temperature in H_of_transformations:
            if lower_temperature <= temperature <= upper_temperature:
                total_transformation_entropy += \
                    H_of_transformations[temperature] / temperature

    return total_transformation_entropy


def entropyCrystalFun(Cp_crystal_fun, H_transformations=None):
    '''
    Generate a function to compute the entropy of a crystal.

    Parameters
    ----------
    Cp_crystal_fun : callable
        Function with one argument for the temperature (in Kelvin) that returns
        the heat capacity at constant pressure for the crystalline phase. It is
        necessary that Cp_crystal_fun returns finite heat capacity values
        between zero Kelvin and any temperature that will be considerade to
        compute the entropy of the crystal.

    H_transformations : dictionary or None or False
        If there are any crystalline phase transitions in the material, this
        information must be given as a dictionary were the keys are the
        temperature where the transition happens (in Kelvin) and the values are
        the enthalpy of transformations (in J/mol) during heating. See the
        documentation for the function totalTransformationEntropy for more
        details. Default value is None.

    Returns
    -------
    ScrystalFun : callable
        Function with one argument for temperature (in Kelvin) that returns the
        entropy of the crystalline phase at the given temperature. The argument
        of the function can be a scalar or an array-like.

    '''
    def fun(x):
        return Cp_crystal_fun(x) / x

    @np.vectorize
    def ScrystalFun(T):

        sum_transformation_entropy = \
            totalTransformationEntropy(H_transformations, 0, T)
        integral = quad(fun, 0, T, limit=100)[0]
        crystal_entropy = integral + sum_transformation_entropy

        return crystal_entropy

    return ScrystalFun


def entropyGlassFun(Cp_glass_fun, glass_entropy_0K=0):
    '''
    Generate a function to compute the entropy of a glass.

    Parameters
    ----------
    Cp_glass_fun : callable
        Function with one argument for the temperature (in Kelvin) that returns
        the heat capacity at constant pressure for the glassy phase. It is
        necessary that Cp_glass_fun returns finite heat capacity values between
        zero Kelvin and any temperature that will be considerade to compute the
        entropy of the glass.

    glass_entropy_0K : integer or float
        The residual entropy of a glass at zero Kelvin. There are some debate
        in the literature about which is the residual entropy of a glass at
        this temperature. The default value is zero.

    Returns
    -------
    SglassFun : callable
        Function with one argument for temperature (in Kelvin) that returns the
        entropy of the glassy phase at the given temperature. The argument of
        the function can be a scalar or an array-like.

    '''
    def fun(x):
        return Cp_glass_fun(x) / x

    @np.vectorize
    def SglassFun(T):
        integral = quad(fun, 0, T, limit=100)[0]
        glass_entropy = integral + glass_entropy_0K

        return glass_entropy

    return SglassFun


def entropyLiquidFun(Cp_liquid_fun, Tm, delta_Hm, entropy_crystal_at_Tm):
    '''
    Generate a function to compute the entropy of a liquid.

    Parameters
    ----------
    Cp_liquid_fun : callable
        Function with one argument for the temperature (in Kelvin) that returns
        the heat capacity at constant pressure for the liquid phase.

    Tm : integer or float
        Melting temperature.

    delta_Hm : integer or float
        Enthalpy of melting.

    entropy_crystal_at_Tm : integer or float
        Entropy of the crystalline phas at the melting temperature right before
        the phase transition to the liquid.

    Returns
    -------
    SliquidFun : callable
        Function with one argument for temperature (in Kelvin) that returns the
        entropy of the liquid phase at the given temperature. The argument of
        the function can be a scalar or an array-like.

    '''
    delta_Sm = delta_Hm / Tm

    def fun(x):
        return Cp_liquid_fun(x) / x

    @np.vectorize
    def SliquidFun(T):
        integral = quad(fun, Tm, T, limit=100)[0]
        liquid_entropy = entropy_crystal_at_Tm + delta_Sm + integral

        return liquid_entropy

    return SliquidFun


def excessEntropyFun(Cp_crystal_fun,
                     Cp_liquid_fun,
                     Tm,
                     delta_Hm,
                     H_transformations=None):
    '''
    Generate a function to compute the excess entropy.

    The excess entropy is the entropy of the liquid phase minus the entropy of
    the crystalline phase.

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
    SexcFun : callable
        Function with one argument for temperature (in Kelvin) that returns the
        excess entropy at the given temperature. The excess entropy is the
        difference between the liquid and the crystalline entropy at the given
        temperature The argument of the function can be a scalar or an
        array-like.

    '''
    delta_Sm = delta_Hm / Tm

    def fun(x):
        return (Cp_liquid_fun(x) - Cp_crystal_fun(x)) / x

    @np.vectorize
    def SexcFun(T):

        sum_transformation_entropy = \
            totalTransformationEntropy(H_transformations, T, Tm)
        integral = quad(fun, Tm, T, limit=100)[0]
        excess_entropy = delta_Sm + integral + sum_transformation_entropy

        return excess_entropy

    return SexcFun


def KauzmannTemperature(Cp_crystal_fun,
                        Cp_liquid_fun,
                        Tm,
                        delta_Hm,
                        H_transformations=None,
                        return_Sexc_fun=False):
    '''
    Computes the Kauzmann temperature where the excess entropy is zero.

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

    return_Sexc_fun : boolean
        If True then the excess entropy function is also returned. Default
        value is False.

    Returns
    -------
    TK : float or nan
        If there is a Kauzmann temperature then this function returns its value
        as a float. If no Kauzmann temperature is found, then the function
        returns nan.

    SexcFun : callable
        Only returned if 'return_Sexc_fun' is True. Function with one argument
        for temperature (in Kelvin) that returns the excess entropy at the
        given temperature. The excess entropy is the difference between the
        liquid and the crystalline entropy at the given temperature The
        argument of the function can be a scalar or an array-like.

    Notes
    -----
    There may be more than one temperature where the excess entropy is zero,
    depending on the crystalline phase transitions that are present. This
    function returns the highest Kauzmann temperature if there is more than one.

    References
    ----------
    [1] Kauzmann, W. (1948). The nature of the glassy state and the behavior of
        liquids at low temperatures. Chemical Reviews 43, 219–256.

    [2] Debenedetti, P.G., and Stillinger, F.H. (2001). Supercooled liquids and
        the glass transition. Nature 410, 259–267.

    '''
    SexcFun = excessEntropyFun(Cp_crystal_fun, Cp_liquid_fun, Tm, delta_Hm,
                               H_transformations)

    temperatures = np.linspace(1, Tm, max(15, int(Tm / 100)))

    if Tm > 300:
        temperatures = np.append(temperatures, 300)
        temperatures = np.sort(temperatures)

    if H_transformations:
        Tlist = []
        for transformation_T in H_transformations:
            Tlist.append(transformation_T - 1)
            Tlist.append(transformation_T + 1)

        temperatures = np.append(temperatures, Tlist)

    temperatures = np.sort(temperatures)

    Sexc = SexcFun(temperatures)
    logic1 = np.isfinite(Sexc)
    logic2 = Sexc != 0
    logic = np.logical_and(logic1, logic2)
    temperatures, Sexc = temperatures[logic], Sexc[logic]

    sign_change = np.where(np.diff(np.sign(Sexc)) != 0)[0]

    if len(sign_change) > 0:
        index = sign_change[-1]
        TK = brentq(SexcFun, temperatures[index], temperatures[index + 1])
    else:
        TK = np.nan

    if return_Sexc_fun:
        return TK, SexcFun
    else:
        return TK
