#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

from .manipulate import removeColumnsWithOnlyZerosMultiIndex

__cur_path = os.path.dirname(__file__)
SCIGLASS_DATABASE_PATH = os.path.join(__cur_path, 'datafiles/sciglass.csv.xz')
SCIGLASS_COMP_DATABASE_PATH = os.path.join(__cur_path, 'datafiles/sciglass_comp.csv.xz')
SCIGLASS_ATFRAC_DATABASE_PATH = os.path.join(__cur_path, 'datafiles/sciglass_atfrac.csv.xz')


def sciglass(load_compounds=False, load_atomic_fraction=True):
    """Load SciGlass data into a pandas DataFrame

    SciGlass is a database of glass properties Copyright (c) 2019 EPAM Systems
    and licensed under ODC Open Database License (ODbL). The database is hosted
    on GitHub [1]. A portion of the SciGlass database is shipped with GlassPy,
    so no additional downloads are necessary.

    This function returns a MultiIndex pandas DataFrame. The first-level
    indexes are:
        at_frac : relative to the atomic fraction of the chemical elements that
            make the glass. Only available if "load_atomic_fraction" is True.

        comp : relative to the chemical compounds that make the glass. Only
            available if "load_compounds" is True.

        meta : metadata.

        prop : properties.

    The property column names are:

        RefractiveIndex : refractive index measured at wavelenght of 589.3 nm.
            Dimensionless.

        AbbeNumber : Abbe number. Dimensionless.

        CTE : linear coefficient of thermal expansion below the glass transition
            temperature. Unit: K^{-1}.

        ElasticModulus : Elastic of Young's Modulus. Unit: GPa.

        Tg : glass transition temperature. Unit: K.

        Tliquidus: liquidus temperature. Unit: K.

        T0 to T12 : "Tn" is the temperature where the base-10 logarithm of
            viscosity (in Pa.s) is "n". Example: T4 is the temperature where
            log10(viscosity) = 4. Unit: K.

        ViscosityAt773K to ViscosityAt2473K : value of base-10 logarithm of
            viscosity (in Pa.s) at a certain temperature. Example:
            ViscosityAt1073K is the log10(viscosity) at 1073 Kelvin.
            Dimensionless.

    Parameters
    ----------
    load_compounds : bool, False
        If True then chemical compounds are loaded and added to the DataFrame

    load_atomic_fraction : bool, True
        If True then the atomic fractions are loaded and added to the DataFrame

    Returns
    -------
    data : pandas DataFrame
        MultiIndex DataFrame containing a portion of the SciGlass database.

    References
    ----------
    [1] Epam/SciGlass. 2019. EPAM Systems, 2019.
        https://github.com/epam/SciGlass.

    """
    data = pd.read_csv(SCIGLASS_DATABASE_PATH, index_col=0)
    metadata_index = ['ChemicalAnalysis', 'Author', 'Year']
    property_index = np.array(sorted(set(data.columns) - set(metadata_index)))
    d = {}

    if load_atomic_fraction:
        data_af = pd.read_csv(SCIGLASS_ATFRAC_DATABASE_PATH, index_col=0)
        data['NumberChemicalElements'] = data_af.astype('bool').sum(axis=1)
        metadata_index.append('NumberChemicalElements')
        d['at_frac'] = data_af

    if load_compounds:
        data_c = pd.read_csv(SCIGLASS_COMP_DATABASE_PATH, index_col=0)
        data['NumberCompounds'] = data_c.astype('bool').sum(axis=1)
        metadata_index.append('NumberCompounds')
        d['comp'] = data_c

    d['meta'] = data[metadata_index]
    d['prop'] = data[property_index]
    data = pd.concat(d, axis=1)

    return data


def sciglassOxides(
        minimum_fraction_oxygen=0.3,
        elements_to_remove=['S', 'H', 'C', 'Pt', 'Au', 'F', 'Cl', 'N', 'Br', 'I'],
        load_compounds=False,
):
    '''Load only the oxides from SciGlass database into a pandas DataFrame

    The default settings of this function follow the definion of an oxide glass
    used in [1]. These can be changed with the parameters of the function.

    SciGlass is a database of glass properties Copyright (c) 2019 EPAM Systems
    and licensed under ODC Open Database License (ODbL). The database is hosted
    on GitHub [1]. A portion of the SciGlass database is shipped with GlassPy,
    so no additional downloads are necessary.

    This function returns a MultiIndex pandas DataFrame. The first-level
    indexes are:
        at_frac : relative to the atomic fraction of the chemical elements that
            make the glass. Only available if "load_atomic_fraction" is True.

        comp : relative to the chemical compounds that make the glass. Only
            available if "load_compounds" is True.

        meta : metadata.

        prop : properties.

    The property column names are:

        RefractiveIndex : refractive index measured at wavelenght of 589.3 nm.
            Dimensionless.

        AbbeNumber : Abbe number. Dimensionless.

        CTE : linear coefficient of thermal expansion below the glass transition
            temperature. Unit: K^{-1}.

        ElasticModulus : Elastic of Young's Modulus. Unit: GPa.

        Tg : glass transition temperature. Unit: K.

        Tliquidus: liquidus temperature. Unit: K.

        T0 to T12 : "Tn" is the temperature where the base-10 logarithm of
            viscosity (in Pa.s) is "n". Example: T4 is the temperature where
            log10(viscosity) = 4. Unit: K.

        ViscosityAt773K to ViscosityAt2473K : value of base-10 logarithm of
            viscosity (in Pa.s) at a certain temperature. Example:
            ViscosityAt1073K is the log10(viscosity) at 1073 Kelvin.
            Dimensionless.

    Parameters
    ----------
    minimum_fraction_oxygen : float
        Minimum atomic fraction of oxygen for the glass to be considered an
        oxide. A value between 0 and 1 is expected.

    elements_to_remove : list or 1-d array or False
        Iterable with the chemical elements (strings) that must not be present
        in the glass in. If None then no chemical element is removed. Default
        value is ['S', 'H', 'C', 'Pt', 'Au', 'F', 'Cl', 'N', 'Br', 'I'].

    load_compounds : bool
        If True then chemical compounds are loaded and added to the DataFrame.
        Default value is False

    Returns
    -------
    data : pandas DataFrame
        MultiIndex DataFrame containing a portion of the SciGlass database.

    References
    ----------
    [1] Alcobaça, E., Mastelini, S.M., Botari, T., Pimentel, B.A., Cassar, D.R.,
        de Carvalho, A.C.P. de L.F., and Zanotto, E.D. (2020). Explainable
        Machine Learning Algorithms For Predicting Glass Transition
        Temperatures. Acta Materialia 188, 92–100.

    [2] Epam/SciGlass. 2019. EPAM Systems, 2019.
        https://github.com/epam/SciGlass.
    
    '''
    data = sciglass(load_compounds, load_atomic_fraction=True)
    logic = data['at_frac']['O'] >= minimum_fraction_oxygen
    data = data.loc[data[logic].index]

    if elements_to_remove:
        for el in elements_to_remove:
            logic = data['at_frac'][el] == 0
            data = data.loc[data[logic].index]

    data = removeColumnsWithOnlyZerosMultiIndex(data, 'at_frac')

    if load_compounds:
        data = removeColumnsWithOnlyZerosMultiIndex(data, 'comp')

    return data
