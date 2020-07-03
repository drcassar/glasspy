#!/usr/bin/env python3

def removeColumnsWithOnlyZerosMultiIndex(data, first_index):
    '''Remove columns with only zeros in MultiIndex pandas DataFrames

    Parameters
    ----------
    data : DataFrame
        MultiIndex dataframe

    first_index : string
        Name of the first level index to search for columns with only zeroes.

    Returns
    -------
    data_clean : DataFrame
        DataFrame with columns with only zeroes removed.

    '''
    nonzero_cols_bool = data[first_index].sum(axis=0).astype(bool)
    zero_cols = data[first_index].columns.values[~nonzero_cols_bool]
    drop_cols = [(first_index, col) for col in zero_cols]
    data_clean = data.drop(drop_cols, axis=1)

    return data_clean
