#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

__cur_path = os.path.dirname(__file__)
SCIGLASS_DATABASE_PATH = os.path.join(__cur_path, 'datafiles/sciglass.zip')
SCIGLASS_COMP_DATABASE_PATH = os.path.join(__cur_path, 'datafiles/sciglass_comp.zip')

CHEMICAL_ELEMENTS_SYMBOL = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
    'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
    'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


def sciglass(load_compounds=False):
    """Load SciGlass data into a pandas DataFrame

    TODO: update docstring

    SciGlass is a database of glass properties Copyright (c) 2019 EPAM Systems
    and licensed under ODC Open Database License (ODbL). The database is hosted
    on GitHub [1]. A portion of SciGlass database is shipped with GlassPy, so no
    additional downloads are necessary.

    The pandas DataFrame returned from this function has some columns that are
    related to the chemical composition of the glasses and some columns that are
    related to attributes of the glasses. The index of this DataFrame is a
    combination of the Glass Number and the Publication Code, separated by an
    underline. These two numbers give an unique ID per glass and were defined by
    the creators of SciGlass.

    Chemical composition is in atomic fraction of the chemical elements that are
    present in the glass.

    The attributes are:

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

        num_elements : number of different chemical elements that are present in
            the glass

    Returns
    -------
    sg_data : pandas DataFrame
        DataFrame containing a portion of the SciGlass database.

    composition_column_names : list
        List containing all the column names related to the composition of the
        glasses. Composition is in atomic fraction.

    attributes_column_names : list
        List containing all the column names related to attributes of the
        glasses.

    References
    ----------
    [1] Epam/SciGlass. 2019. EPAM Systems, 2019.
        https://github.com/epam/SciGlass.

    """
    data = pd.read_csv(SCIGLASS_DATABASE_PATH, index_col=0)

    # Composition
    columns_set = set(data.columns)
    composition_column_names = \
        np.array(sorted(columns_set.intersection(CHEMICAL_ELEMENTS_SYMBOL)))

    # Property
    columns_set = set(data.columns)
    property_column_names = \
        np.array(sorted(columns_set - set(composition_column_names)))

    # Features
    data['NumberChemicalElements'] = \
        data[composition_column_names].astype('bool').sum(axis=1)
    feature_column_names = ['NumberChemicalElements', 'ChemicalAnalysis']

    # Compounds
    if load_compounds:
        cdata = pd.read_csv(SCIGLASS_COMP_DATABASE_PATH, index_col=0)
        compounds_column_names = cdata.columns.values.tolist()
        data['NumberCompounds'] = \
            cdata[compounds_column_names].astype('bool').sum(axis=1)
        feature_column_names.append('NumberCompounds')
        
        d = {
            'at_frac': data[composition_column_names],
            'comp': cdata,
            'feat': data[feature_column_names],
            'prop': data[property_column_names],
            }

    else:
        d = {
            'at_frac': data[composition_column_names],
            'feat': data[feature_column_names],
            'prop': data[property_column_names],
            }

    data = pd.concat(d, axis=1)

    return data


def sciglassOxides(
        minimum_fraction_oxygen=0.3,
        elements_to_remove=['S', 'H', 'C', 'Pt', 'Au', 'F', 'Cl', 'N', 'Br', 'I'],
        load_compounds=False,
):
    '''Load only the oxides from SciGlass database into a pandas DataFrame

    TODO: update docstring

    The default settings of this function follow the definion of an oxide glass
    used in [1]. These can be changed with the parameters of the function.

    SciGlass is a database of glass properties Copyright (c) 2019 EPAM Systems
    and licensed under ODC Open Database License (ODbL). The database is hosted
    on GitHub [2]. A portion of SciGlass database is shipped with GlassPy, so no
    additional downloads are necessary.

    The pandas DataFrame returned from this function has some columns that are
    related to the chemical composition of the glasses and some columns that are
    related to attributes of the glasses. The index of this DataFrame is a
    combination of the Glass Number and the Publication Code, separated by an
    underline. These two numbers give an unique ID per glass and were defined by
    the creators of SciGlass.

    Chemical composition is in atomic fraction of the chemical elements that are
    present in the glass.

    The attributes are:

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

        num_elements : number of different chemical elements that are present in
            the glass

    Parameters
    ----------
    minimum_fraction_oxygen : float
        Minimum atomic fraction of oxygen for the glass to be considered an
        oxide. A value between 0 and 1 is expected.

    elements_to_remove : list or 1-d array
        Iterable with the chemical elements (strings) that must not be present
        in the glass in. If None then no chemical element is removed.

    Returns
    -------
    sg_data : pandas DataFrame
        DataFrame containing a portion of the oxide glasses in SciGlass database. 

    composition_column_names : list
        List containing all the column names related to the composition of the
        glasses. Composition is in atomic fraction.

    attributes_column_names : list
        List containing all the column names related to attributes of the
        glasses.

    References
    ----------
    [1] Alcobaça, E., Mastelini, S.M., Botari, T., Pimentel, B.A., Cassar, D.R.,
        de Carvalho, A.C.P. de L.F., and Zanotto, E.D. (2020). Explainable
        Machine Learning Algorithms For Predicting Glass Transition
        Temperatures. Acta Materialia 188, 92–100.

    [2] Epam/SciGlass. 2019. EPAM Systems, 2019.
        https://github.com/epam/SciGlass.
    
    '''
    data = sciglass(load_compounds)

    logic = data['at_frac']['O'] >= minimum_fraction_oxygen
    data = data[logic]

    if elements_to_remove:
        for el in elements_to_remove:
            logic = data['at_frac'][el] == 0
            data = data[logic]

    # Removing obsolete chemical element columns
    nonzero_cols_bool = data['at_frac'].sum(axis=0).astype(bool)
    zero_cols = data['at_frac'].columns.values[~nonzero_cols_bool]

    data = data.swaplevel(axis=1)
    data.drop(zero_cols, axis=1, inplace=True)
    data = data.swaplevel(axis=1)

    return data
