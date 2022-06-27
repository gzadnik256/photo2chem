from operator import index
import os
from typing import List

import numpy as np

import pandas as pd


def load_data(
        data_dir='data',
        filename='masterapogeematch.csv',
        url='http://fiz.fmf.uni-lj.si/zwitter/PRODEX/masterapogeematch.csv'
    ):
    """Load the CSV into the DataFrame."""
    full_path = os.path.join(data_dir, filename)
    if not os.path.exists(full_path):  # if file is not on the disk already
        if not os.path.exists(data_dir):  # if data dir does not exist yet
            os.makedirs(data_dir)  # create data dir
        import urllib.request
        urllib.request.urlretrieve(url, full_path)  # download the data
    return pd.read_csv(full_path, header=1, index_col=False)


def clean_data(df):
    """Remove unused columns, add missing columns, make NA values standardized."""
    # clean the data
    df.drop(df.columns[-1], axis=1, inplace=True)  # last column is empty when loaded
    df.columns = [x.strip(' |#') for x in df.columns]  # clean the column names

    # Add month and day field so that column numbers are aligned with the doc
    c = 2
    if 'month' not in df.columns:
        df['month'] = df.shape[0] * [0]
        c -= 1
    if 'day' not in df.columns:
        df['day'] = df.shape[0] * [0]
        c -= 1
    # Appropriately order columns
    df = df[['inst', 'year', 'month', 'day'] + [df.columns[x] for x in range(2+c, 74+c)]]

    # Make NaN what is NaN
    df.loc[df['ak']==-99, 'ak'] = np.nan
    df.loc[df['kmag']==-99, 'kmag'] = np.nan
    df.loc[df['ekmag']==-99, 'ekmag'] = np.nan
    df.loc[df['ejmag']==-99, 'ejmag'] = np.nan
    df.loc[df['pmra']==-9999, 'pmra'] = np.nan
    df.loc[df['epmde']==-99, 'epmde'] = np.nan
    df.loc[df['epmra']==-99, 'epmra'] = np.nan

    # Zero measured b-mag
    # May also be filtered by Nb = 0, Nc = 0
    df.loc[df['magbr2']==0, 'magbr2'] = np.nan
    df.loc[df['magbr3']==0, 'magbr3'] = np.nan
    df.loc[df['magbr4']==0, 'magbr4'] = np.nan
    
    # Zero measured c-mag
    df.loc[df['magcr2']==0, 'magcr2'] = np.nan
    df.loc[df['magcr3']==0, 'magcr3'] = np.nan
    df.loc[df['magcr4']==0, 'magcr4'] = np.nan

    return df


def get_magnitude_diffs(df, ordered_mag_columns):
    """Get consecutive diffs of the columns listed in ordered_mag_columns param."""
    df_diffs = pd.DataFrame(index=df.index)
    for m0, m1 in zip(ordered_mag_columns, ordered_mag_columns[1:]):
        df_diffs[f'{m0}_{m1}'] = df[m0] - df[m1]
    return df_diffs


def filter_rows_by_std(df, df_std, std_thresholds: dict):
    """Filter rows of a df based on df_std rows and thresholds set in std_thresholds.
    
    Usecase: df_std is a dataframe of standard deviations of measurements in df. One would
    like to skip the measurements in df that has standard deviation (for a specific measurement)
    over the threshold (which is set in std_thresholds dict).
    
    :param df: dataframe of measurements
    :param df_std: dataframe containing at least columns htat are present as keys on std_thresholds
    :param std_thresholds: a dict of (column_name, max_acceptable_value) key-value pairs.
    """
    assert all(df.index == df_std.index)
    return df[list(map(all, np.transpose(np.array([df_std[k] < std_thresholds[k] for k in std_thresholds.keys()]))))]


def join_rows_from_raw(filtered_df: pd.DataFrame, raw_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Append cols in the list from raw_df to filtered_df.
    
    Have in mind that filtered_df.index is a subset of raw_df.index.
    """
    # Another way:
    # raw_df[cols].loc[filtered_df.index]
    filtered_df_copy = filtered_df.copy()
    for col in cols:
        # can only select filtered_df.index rows from raw_df[col] is possible because raw_df[col] is pd.Series;
        # in case of dataframe, should set raw_df[cols].loc[filtered_df.index]
        filtered_df_copy[col] = raw_df[col][filtered_df_copy.index]
    if filtered_df_copy.isnull().values.any():
        print(f'NA\'s appear in the extended dataframe in columns '
              f'{filtered_df_copy.columns[list(filtered_df_copy.isnull().any(axis=0))]}')
    # TODO what to do with NAs, if appear ???
    return filtered_df_copy


# PREDICTORS & TARGET 
#####################

# Skip : 33 - 36 (including) -- positions
#        71 - 75 (including) -- related to the positions
# but may include 73 - 75 in evaluations

# skipped also: 0 which telescope, 1-3 date fields, 4 field, 64 gaiaid, 48 text_flag and 28 apogeeid (non-numeric)
# TODO use later for filtering / model split ?

predictors = [
    'Na', 'Nb', 'Nc', 'magar2', 'sigar2', 'magbr2', 'sigbr2', 'magcr2',
    'sigcr2', 'magar3', 'sigar3', 'magbr3', 'sigbr3', 'magcr3', 'sigcr3',
    'magar4', 'sigar4', 'magbr4', 'sigbr4', 'magcr4', 'sigcr4', 'jmag',
    'ejmag', 'kmag', 'ekmag', 'teffflag', 'loggflag', 'vmicroflag',
    'mhflag', 'amflag', 'vsiniflag', 'cafeflag', 'fehflag', 'ak', 'ebvsfd',
    'pmra', 'epmra', 'pmde', 'epmde', 'gmag', 'bpmag', 'rpmag'
]
targets = ['cafe', 'feh']

# Predictor columns got from the dataframe df constructed as
# - loading CSV
# - cleaning it
# as
# df.columns[
#    # [0] +
#    [x for x in range(7, 28)] +  # 5, 6: xx, yy
#    [x for x in range(29, 33)] + 
#    [x for x in range(49, 55)] + 
#    [x for x in range(59, 63)] +  # 63: gaiaid
#    [x for x in range(64, 71)]
#]
