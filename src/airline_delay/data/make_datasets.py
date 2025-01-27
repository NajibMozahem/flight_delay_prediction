# Copyright (c) 2024 Houssem Ben Braiek, Emilio Rivera-Landos, IVADO, SEMLA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""This module includes the functions to make datasets for bank marketing ML applications.

Examples:
    >>> from bank_marketing.data.make_datasets import make_bank_marketing_dataframe
    >>> bank_db_file = r"/path/to/bank_marketing.db"
    >>> socio_eco_data_file = r"/path/to/socio_economic_indices_data.csv"
    >>> df = make_bank_marketing_dataframe(bank_db_file, socio_eco_data_file)
"""

import logging
import os
from typing import Tuple, Dict
import fsspec
import fsspec.implementations.local
import pandas as pd

# from bank_marketing.sqlite_db.bank_marketing_DAL import BankMarketingDAL
# from bank_marketing.helpers.file_loaders import load_fsspec_locally_temp

logger = logging.getLogger('bank_marketing.data.make_datasets')


def extract_credit_features(row: pd.Series) -> Tuple[str, str]:
    """Deduce if the customer has any credit or any default of payment.

    Based on two columns: status and default penalites that are present
    in loans and mortgages data tables.

    Args:
        row (pd.Series): mortgage/loan entry (row) for one customer

    Returns:
        Tuple[str, str]: has loan (yes/no/unknown), had default (yes/no/unknown)
    """
    loan, default = None, None
    if row['status'] == 'paid':
        loan = 'no'
    elif row['status'] == 'ongoing':
        loan = 'yes'
    elif row['status'] == 'unknown':
        loan = 'unknown'
    if row['default_penalties'] != row['default_penalties'] or not (row['default_penalties']):
        default = 'unknown'
    elif row['default_penalties'] == 0:
        default = 'no'
    elif row['default_penalties'] > 0:
        default = 'yes'
    return loan, default


def merge_defaults(row: pd.Series) -> str:
    """Merge two default columns resulting from the fusion of loans and mortgages dataframes.

    Args:
        row (pd.Series): entry (row) for one customer aggregated data

    Returns:
        str: has default overall (yes/no/unknown)
    """
    if row['default_x'] == 'yes' or row['default_y'] == 'yes':
        return 'yes'
    elif row['default_x'] == 'unknown' or row['default_y'] == 'unknown':
        return 'unknown'
    elif row['default_x'] == 'no' and row['default_y'] == 'no':
        return 'no'
    return None


def make_airline_dataframe(
    db_file: os.PathLike
) -> pd.DataFrame:
    """Extract data from data file.

    Args:
        db_file (os.PathLike): airline data file path

    Returns:
        pd.DataFrame: airline dataframe 
    """
    logger.info('Building the airline dataframe')
    try:
        with fsspec.open(db_file) as f:
            dataframe = pd.read_csv(f)
        logger.info('Loaded airline data with %d rows', dataframe.shape[0])
    except Exception:
        logger.exception('Failed to load airline data')
        raise

    return dataframe

def make_dataframe_splits_bimonthly(df, date_column:str) -> Dict:
    """
    Splits a DataFrame into bimonthly segments based on a specified date column 
    and saves the segments as CSV files in the provided directory.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be split.
    - date_column (str): The column containing date information for segmentation.
    """
    splits_dict = {}
    month_year_series = df[date_column].apply(lambda date:'-'.join(date.split('-')[:2]))
    month_year_list = list(set(month_year_series.tolist()))
    sorted_month_year_list = sorted(month_year_list, key=lambda myp: tuple(myp.split('-')))
    for idx in range(0, len(sorted_month_year_list)-1, 2):
        myp1 = sorted_month_year_list[idx]
        data_at_myp1 = df[df[date_column].apply(lambda date:myp1 in date)]
        myp2 = sorted_month_year_list[idx+1]
        data_at_myp2 = df[df[date_column].apply(lambda date:myp2 in date)]
        
        data_to_myp2 = pd.concat([data_at_myp1, data_at_myp2], axis=0)
        splits_dict[myp2] = data_to_myp2
    return splits_dict