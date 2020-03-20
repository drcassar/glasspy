#!/usr/bin/env python3

import pandas as pd
import os

__cur_path = os.path.dirname(__file__)
SCIGLASS_DATABASE_PATH = os.path.join(__cur_path, 'datafiles/sciglass.zip')

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


def sciglass():
    """Load SciGlass data into a pandas DataFrame

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
        glasses. Composition information is in atomic fraction.

    attributes_column_names : list
        List containing all the column names related to attributes of the
        glasses.

    References
    ----------
    [1] Epam/SciGlass. 2019. EPAM Systems, 2019.
        https://github.com/epam/SciGlass.

    """
    sg_data = pd.read_csv(SCIGLASS_DATABASE_PATH, index_col=0)
    columns_set = set(sg_data.columns)
    composition_column_names = \
        list(sorted(columns_set.intersection(CHEMICAL_ELEMENTS_SYMBOL)))
    sg_data['num_elements'] = \
        sg_data[composition_column_names].astype('bool').sum(axis=1)
    columns_set = set(sg_data.columns)
    attributes_column_names = \
        list(sorted(columns_set - set(composition_column_names)))

    return sg_data, composition_column_names, attributes_column_names
