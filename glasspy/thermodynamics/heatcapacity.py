import numpy as np


def _getCFunction(info):

    if info is None:

        def Cfun(T):
            return np.nan

    elif isinstance(info, dict):

        T0 = info.get('T0', 0)
        T1 = info.get('T1', 0)
        T2 = info.get('T2', 0)
        T3 = info.get('T3', 0)
        T4 = info.get('T4', 0)
        _T1 = info.get('_T1', 0)
        _T2 = info.get('_T2', 0)
        _T3 = info.get('_T3', 0)
        _T1_2 = info.get('_T1_2', 0)

        def Cfun(T_):
            T = float(T_)
            return T0 + T1 * T + T2 * T**2 + T3 * T**3 + T4 * T**4 \
                + _T1 * T**(-1) + _T2 * T**(-2) + _T3 * T**(-3) \
                + _T1_2 * T**(-1/2)

    else:
        raise TypeError('A dictionary or None must be provided as an argument')

    return Cfun


def heatCapacityFunFromTuple(C_tuple, extrapolate_borders=False):
    '''
    Generate a heat capacity function from a heat capacity tuple.

    Parameters
    ----------
    C_tuple : tuple
        Heat capacity tuple. See notes for the expected format.

    extrapolate_borders : True, False, 'both', 'upper', or 'lower'
        Parameter configuring if the generated function will extrapolate or not
        the functions in the C_tuple. If 'False' then no extrapolation will be
        made. If 'True' or 'both', then both temperature ends will be
        extrapolated if necessary. If 'lower' then only the heat capacity for
        lower temperatures will be extrapolated. If 'upper' then only the heat
        capacity for higher temperatures will be extrapolated. The default
        value is 'False'.

    Returns
    -------
    Cfun : callable
        Python function to compute the heat capacity following the information
        provided in C_tuple. This function accepts one argument named 'T' for
        temperature. 'T' can be any finite number or an array-like object. If
        the heat capacity cannot be computed (see 'extrapolate_borders') a nan
        value is returned.

    Notes
    -----
    The expected C_tuple format is a tuple of tuples. Below is a simple example
    for the heat capacity of liquid B2O3:

        Cp_tuple_B2O3_liquid = (
            (0, None),
            (723, {
                'T0': 128.3,
                'T1': -3.01E-4,
            }, 'liquid'),
            (3200, None),
        )

    Observe that the tuple Cp_tuple_B2O3_liquid has three tuples inside, each
    of these tuples with two or three items. These tuples inside the main tuple
    will be called subtuples from now on.

    The first item of these subtuples must be a float, indicating the lowest
    temperature limit of valitity of the heat capacity function of the
    subtuple. It is expected that the first subtuple in any C_tuple starts with
    zero Kelvin, as this is the lowest temperature where we can measure heat
    capacity. The upper temperature limit is always defined by the first item
    in the next subtuple in the sequence.

    The second item of the subtuples can be None or a dictionary with
    information about the heat capacity function. If the second item is None,
    then there is no heat capacity function defined. If the second item is a
    dictionary, then the keys of the dictionary indicate which coefficients of
    the heat capacity function are present, and the values of the dictionary
    indicate the values of the coefficients. In the example, in the range of
    723 to 3200 Kelvin, there are two coefficients, namely 'T0' and 'T1'. In
    this particular case, the heat capacity function is equivalent to:

    def function(T):
        return -3.01e-4*T + 128.3

    The valid coefficient names are:

        'T0': coefficient independent of temperature
        'T1': coefficient that multiplies T
        'T2': coefficient that multiplies T**2
        'T3': coefficient that multiplies T**3
        'T4': coefficient that multiplies T**4
        '_T1': coefficient that multiplies T**(-1)
        '_T2': coefficient that multiplies T**(-2)
        '_T3': coefficient that multiplies T**(-3)
        '_T1_2': coefficient that multiplies T**(-1/2)

    These are the usual coefficients found in the literature. More coefficients
    can be easily added by modifying '_getCFunction'.

    Finally, the third and last item of the subtuple is optional. It can hold
    any information that the user wants to store. It is never checked in any
    function of GlassPy.

    References
    ----------
    [1] Maier, C.G., and Kelley, K. (1932). An equation for the representation
        of high-temperature heat content data. Journal of the American Chemical
        Society 54, 3243–3246.

    '''
    functions = []

    if extrapolate_borders is False:
        for i in range(len(C_tuple)):
            segment_function = _getCFunction(C_tuple[i][1])
            functions.append(segment_function)

    else:
        if C_tuple[0][1] is None and (
                extrapolate_borders is True
                or extrapolate_borders in ['lower', 'both']):
            segment_function = _getCFunction(C_tuple[1][1])
            functions.append(segment_function)
        else:
            segment_function = _getCFunction(C_tuple[0][1])
            functions.append(segment_function)

        for i in range(len(C_tuple) - 2):
            segment_function = _getCFunction(C_tuple[i + 1][1])
            functions.append(segment_function)

        if C_tuple[-1][1] is None and (
                extrapolate_borders is True
                or extrapolate_borders in ['upper', 'both']):
            segment_function = _getCFunction(C_tuple[-2][1])
            functions.append(segment_function)
        else:
            segment_function = _getCFunction(C_tuple[-1][1])
            functions.append(segment_function)

    @np.vectorize
    def Cfun(T):
        if T < 0:
            return np.nan

        elif T == 0:
            return 0

        else:
            for i in range(len(C_tuple) - 1):
                if T > C_tuple[i][0] and T <= C_tuple[i + 1][0]:
                    return functions[i](T)

            return functions[-1](T)

    return Cfun
