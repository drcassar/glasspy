#!/usr/bin/env python3

import pandas as pd

SCIGLASS_DATABASE_PATH = r'../../data/sciglass.zip'

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
    on GitHub at https://github.com/epam/SciGlass. A portion of SciGlass
    database is shipped with GlassPy, so no additional downloads are necessary.

    Returns
    -------
    sg_data : pandas DataFrame
        DataFrame containing a portion of the SciGlass database.

    composition_column_names : list
        List containing all the column names related to the composition of the
        glasses. Composition information is in atomic fraction.

    attributes_column_names : list
        List containing all the column names related to attributes of the glasses.
        TODO: units and definition
        TODO: explain what the index is

    """

    sg_data = pd.read_csv(SCIGLASS_DATABASE_PATH, index_col=0)

    columns_set = set(sg_data.columns)

    composition_column_names = \
        list(sorted(columns_set.intersection(CHEMICAL_ELEMENTS_SYMBOL)))

    attributes_column_names = \
        list(sorted(columns_set - set(composition_column_names)))

    return sg_data, composition_column_names, attributes_column_names


def sciglassQuery():
    pass
