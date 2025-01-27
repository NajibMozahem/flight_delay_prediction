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
"""Exploration and analysis of data distribution.

This module includes the functions to explore and analyze columns' data distributions
in a dataframe for airline ML applications.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prettytable import PrettyTable


def report_columns_inspection(
    data_frame: pd.DataFrame, columns: Optional[List[str]] = None, decimals: int = 2
) -> None:
    """Print dataframe column inspections results.

    Args:
        data_frame (pd.DataFrame): Data
        columns (List[str], optional): columns to be involved in inspection.
                                    Defaults to None (all columns will be inspected).
        decimals (int, optional): number decimal digits of precision. Defaults to 2.
    """
    numerical_cols_insp_results, categorical_cols_insp_results = inspect_columns(
        data_frame, columns
    )
    if numerical_cols_insp_results:
        print_numerical_columns_inspection(numerical_cols_insp_results, decimals)
    if categorical_cols_insp_results:
        print_categorical_columns_inspection(categorical_cols_insp_results, decimals)


def inspect_columns(
    data_frame: pd.DataFrame, columns: Optional[List[str]] = None
) -> Tuple[Dict, Dict]:
    """Inspect dataframe, specifically, the provided list of columns.

    The results will be encoded as two dictionaries:
    - numerical_cols_insp_results: This dictionary contains inspection results
    for numerical columns. It includes summary statistics obtained using the describe()
    method for each numerical column.
    - categorical_cols_insp_results: This dictionary contains inspection results
    for categorical columns. It includes the count of unique options,
    the unique options themselves, the count of each unique option,
    and the ratio of each unique option's count to the total number of rows.

    Args:
        data_frame (pd.DataFrame): data
        columns (List[str], optional): columns to be inspected. Defaults to None (all).

    Returns:
        Tuple[Dict,Dict]: inspection results dictionaries for numerical and categorical columns
    """
    numerical_cols_insp_results, categorical_cols_insp_results = {}, {}
    columns = data_frame.columns if columns is None else columns
    for col_name in columns:
        col = data_frame[col_name]
        if col.dtype == 'object':
            value_counts = col.value_counts().to_dict()
            categorical_cols_insp_results[col_name] = {
                'options_count': len(col.unique()),
                'options_set': list(col.unique()),
                'values_count': [value_counts[value] for value in list(col.unique())],
                'values_ratio': [
                    value_counts[value] / len(data_frame) for value in list(col.unique())
                ],
            }
        else:
            numerical_cols_insp_results[col_name] = col.describe().to_dict()
    return numerical_cols_insp_results, categorical_cols_insp_results


def print_numerical_columns_inspection(numerical_cols_insp_results: Dict, decimals: int = 2):
    """Print numerical columns inspections results.

    Args:
        numerical_cols_insp_results (Dict): Dictionary of results
        decimals (int, optional): number decimal digits of precision. Defaults to 2.
    """
    assert len(numerical_cols_insp_results) > 0, 'There is no inspected categorical columns'
    tab = PrettyTable()
    numerical_cols = list(numerical_cols_insp_results.keys())
    tab.field_names = ['Column'] + list(numerical_cols_insp_results[numerical_cols[0]].keys())
    tab.add_rows(
        [
            [col] + np.round(list(numerical_cols_insp_results[col].values()), decimals).tolist()
            for col in numerical_cols
        ]
    )
    print(tab)


def print_categorical_columns_inspection(categorical_cols_insp_results: Dict, decimals: int = 2):
    """Print categorical columns inspections results.

    Args:
        categorical_cols_insp_results (Dict): Dictionary of results
        decimals (int, optional): number decimal digits of precision. Defaults to 2.
    """
    assert len(categorical_cols_insp_results) > 0, 'There is no inspected categorical columns'
    tab = PrettyTable()
    categorical_cols = list(categorical_cols_insp_results.keys())
    tab.field_names = ['Column'] + list(categorical_cols_insp_results[categorical_cols[0]].keys())
    for col_name in categorical_cols:
        col_insp_result = categorical_cols_insp_results[col_name]
        options_counts = col_insp_result['options_count']
        tab.add_row(
            [
                col_name,
                options_counts,
                col_insp_result['options_set'][0],
                col_insp_result['values_count'][0],
                round(col_insp_result['values_ratio'][0], decimals),
            ]
        )
        for i in range(1, options_counts):
            tab.add_row(
                [
                    '-',
                    '-',
                    col_insp_result['options_set'][i],
                    col_insp_result['values_count'][i],
                    round(col_insp_result['values_ratio'][i], decimals),
                ],
                divider=(i == options_counts - 1),
            )

    print(tab)
