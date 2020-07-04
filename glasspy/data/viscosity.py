import numpy as np
import pandas as pd
import os
from chemparse import parse_formula

from .load import sciglass


__cur_path = os.path.dirname(__file__)
VISCOSITY_AT_FRAC = os.path.join(__cur_path, 'datafiles/viscosity_at_frac.csv.xz')
VISCOSITY_COMPOUND = os.path.join(__cur_path, 'datafiles/viscosity_compounds.csv.xz')


def genViscosityTable(compounds=False):
    '''Generates a viscosity table from the SciGlass database

    Parameters
    ----------
    compounds : bool, False
        If True, then the chemical compounds are returned, but not the atomic
        fractions. The opposite happens if False.

    Returns
    -------
    data : pandas DataFrame
        DataFrame containing all viscosity data from the SciGlass database.

    '''
    if compounds:
        c_index = 'comp'
        load_compounds = True
        load_atomic_fraction = False
    else:
        c_index = 'at_frac'
        load_compounds = False
        load_atomic_fraction = True

    sg_data = sciglass(
        load_compounds=load_compounds,
        load_atomic_fraction=load_atomic_fraction,
    )

    # Temperatures with a fixed viscosity value
    Tviscos = np.arange(0, 13)
    df_lst = []
    for logvisc in Tviscos:
        d = sg_data.dropna(subset=[('prop', f'T{logvisc}')])
        comp = d[c_index].copy()
        comp['temperature'] = d[('prop', f'T{logvisc}')]
        comp['log_viscosity'] = logvisc
        comp['author'] = d[('meta', 'Author')]
        comp['year'] = d[('meta', 'Year')]
        df_lst.append(comp)
    data1 = pd.concat(df_lst)

    # Viscosity at a fixed temperature
    viscoT = np.arange(0, 1700, 100) + 873
    df_lst = []
    for T in viscoT:
        if f'ViscosityAt{T}K' in d['prop'].columns:
            d = sg_data.dropna(subset=[('prop', f'ViscosityAt{T}K')])
            comp = d[c_index].copy()
            comp['temperature'] = T
            comp['log_viscosity'] = d[('prop', f'ViscosityAt{T}K')]
            comp['author'] = d[('meta', 'Author')]
            comp['year'] = d[('meta', 'Year')]
            df_lst.append(comp)
    data2 = pd.concat(df_lst)

    data = pd.concat([data1, data2], sort=False)
    nonzero_cols_bool = data.sum(axis=0).astype(bool)
    zero_cols = data.columns.values[~nonzero_cols_bool]
    data = data.drop(zero_cols, axis=1)

    return data


def loadViscosityTable(compounds=False):
    '''Loads a viscosity table from the SciGlass database

    This function returns the sabe DataFrame as genViscosityTable, but is
    faster.

    Parameters
    ----------
    compounds : bool, False
        If True, then the chemical compounds are returned, but not the atomic
        fractions. The opposite happens if False.

    Returns
    -------
    data : pandas DataFrame
        DataFrame containing all viscosity data from the SciGlass database.

    '''
    if compounds:
        path = VISCOSITY_COMPOUND
    else:
        path = VISCOSITY_AT_FRAC

    data = pd.read_csv(path, index_col=0)

    return data
        

def viscosityFromString(chemical_formula, decimal_round=3, viscosity_df=None):
    '''Query viscosity data from SciGlass from a string

    Parameters
    ----------
    chemical_formula : string
        String of the chemical formula for the query.

    decimal_round : integer, optional
        Decimal place to round the chemical composition for the query. Default
        value is 3.

    viscosity_df : pandas DataFrame or None
        If a DataFrame is supplied, then it is used for the query. If None then
        the SciGlass viscosity dataset is used. Note that the provided DataFrame
        must be in atomic fraction and have the last four columns with names
        'T', 'log_visc', 'Author', and 'Year'.

    Returns
    -------
    data : pandas DataFrame
        DataFrame with the result of the query. The columns are "temperature"
        for the temperature in Kelvin, "log_viscosity" for the base-10 logarithm
        of viscosity (taken from viscosity measured in Pa.s), "Author" for the
        name of the first author of the original source, and "Year" for the year
        of publication of the original source.

    Info
    ----
    The string for the chemical formula must be an acceptable string for the
    module chemparse [1], examples include simple strings with integers such as
    "SiO2", strings with fractional stoichiometry such as "C1.5O3", and strings
    with non-nested parentheses such as "(LiO2)1(SiO2)2".

    References
    ----------
    [1] https://pypi.org/project/chemparse/
    
    '''
    def parseFormulaNorm(formula, sum_total=1, round_=False):
        dic = parse_formula(formula)
        sum_values = sum(dic.values())

        if round_:
            norm_dic = {k : round(v * sum_total / sum_values, round_)
                        for k,v in dic.items()}
        else:
            norm_dic = {k : v / sum_values for k,v in dic.items()}

        return norm_dic

    non_chem_col = ['temperature', 'log_viscosity', 'author', 'year']

    if viscosity_df is None:
        data = loadViscosityTable(compounds=False)
    else:
        data = viscosity_df.copy()

    atoms = list(set(data.columns.values) - set(non_chem_col))
    data[atoms] = data[atoms].round(decimal_round).copy()

    cdic = parseFormulaNorm(chemical_formula, 1, decimal_round)

    for at in cdic:
        logic = data[at] == cdic[at]
        data = data[logic]

    for at in set(atoms) - set(cdic):
        logic = data[at] == 0
        data = data[logic]

    data = data[non_chem_col]

    return data
