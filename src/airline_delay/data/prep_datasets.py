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
"""This module includes the functions to prepare datasets for ML applications."""

import os
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger('bank_marketing.data.prep_datasets')


@dataclass
class Dataset:
    """A dataclass used to represent a Dataset.

    Attributes:
    ----------
    train_x : Pandas DataFrame
        the dataframe of input features w.r.t training split
    val_x : Pandas DataFrame
        the dataframe of input features w.r.t validation split
    test_x : Pandas DataFrame
        the dataframe of input features w.r.t testing split
    train_y : Pandas Series
        the series of output label w.r.t training split
    val_y : Pandas Series
        tthe series of output label w.r.t validation split
    test_y : Pandas Series
        the series of output label w.r.t testing split
    """

    train_x: pd.DataFrame
    val_x: pd.DataFrame
    test_x: pd.DataFrame
    train_y: pd.Series
    val_y: pd.Series
    test_y: pd.Series

    def merge_in(self, dataset):
        self.train_x = pd.concat([self.train_x, dataset.train_x], axis=0)
        self.val_x = pd.concat([self.val_x, dataset.val_x], axis=0)
        self.test_x = pd.concat([self.test_x, dataset.test_x], axis=0)
        self.train_y = pd.concat([self.train_y, dataset.train_y], axis=0)
        self.val_y = pd.concat([self.val_y, dataset.val_y], axis=0)
        self.test_y = pd.concat([self.test_y, dataset.test_y], axis=0)

    def persist(self, dirpath):
        self.train_x.to_csv(os.path.join(dirpath,'train_x.csv'), sep=';', index=False)
        self.train_y.to_csv(os.path.join(dirpath, 'train_y.csv'), sep=';', index=False)
        self.val_x.to_csv(os.path.join(dirpath, 'val_x.csv'), sep=';', index=False)
        self.val_y.to_csv(os.path.join(dirpath, 'val_y.csv'), sep=';', index=False)
        self.test_x.to_csv(os.path.join(dirpath,'test_x.csv'), sep=';', index=False)
        self.test_y.to_csv(os.path.join(dirpath, 'test_y.csv'), sep=';', index=False)

    # Prints all datasets shapes
    def __repr__(self):
        return f"Dataset(train_x: {self.train_x.shape}, val_x: {self.val_x.shape}, test_x: {self.test_x.shape}, train_y: {self.train_y.shape}, val_y: {self.val_y.shape}, test_y: {self.test_y.shape})"

    def to_markdown(self):
        return f"""
        | Dataset | X | y |
        | - | - | - |
        | Train | {self.train_x.shape} | {self.train_y.shape} |
        | Validation | {self.val_x.shape} | {self.val_y.shape} |
        | Test | {self.test_x.shape} | {self.test_y.shape} |
        """

def preprocess(df):
    orig_count = df.shape[0]
    # remove cancelled flights and diverted flights
    df = df.loc[(df["CANCELLED"]==0)&(df["DIVERTED"]==0)]
    # drop cancellation code column
    df = df.drop(columns=['CANCELLATION_CODE'])
    # remove records with important feature missing
    df = df.dropna(subset = ["AIR_TIME","ARR_TIME","DEP_TIME"])
    print(f"{(100*(orig_count - df.shape[0])/orig_count):.2f} % of data are removed.")
    
    # # convert fl_date to date 
    # if df["FL_DATE"].dtype == "object":
    #     df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format = "%Y-%m-%d")
    
    return df

def prepare_regression_tabular_data(
    data_frame: pd.DataFrame,
    predictors: List[str],
    predicted: str,
    splits_sizes: Optional[List[float]] = None,
    seed: int = 42,
) -> Dataset:
    """Prepare the training/validation/test inputs (X) and outputs (y) for regression modeling.

    Args:
        data_frame (pd.DataFrame): aggregated data frame
        predictors (List[str]): list of predictors column names
        predicted (str): column name of the predicted outcome
        splits_sizes (List[float], optional): list of relative size portions for training, validation, test data, respectively. Defaults to [0.7,0.1,0.2].
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        Dataset: datasets for regression with training/validation/test splits
    """

    data_frame = preprocess(data_frame)
    
    if splits_sizes is None:
        splits_sizes = [0.7, 0.1, 0.2]
    if abs(1 - sum(splits_sizes)) > 1e-3:
        raise ValueError(f'Split sizes must sum to 1.0. Given split sizes: {splits_sizes}')
    if not set(predictors).issubset(data_frame.columns):
        missing_cols = set(predictors) - set(data_frame.columns)
        raise ValueError(f'Predictor columns missing in data frame: {missing_cols}')
    if predicted not in data_frame.columns:
        raise ValueError(f"Predicted column '{predicted}' is missing in data frame.")

    # Remove rows where the target variable is null
    initial_size = len(data_frame)
    data_frame = data_frame.dropna(subset=[predicted])
    rows_dropped = initial_size - len(data_frame)
    if rows_dropped > 0:
        logger.info(f'Dropped {rows_dropped} rows with null values in target variable {predicted}')

    # Handle missing values in predictor variables
    # For numeric columns: fill with median
    numeric_predictors = data_frame[predictors].select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_predictors:
        null_count = data_frame[col].isnull().sum()
        if null_count > 0:
            data_frame[col] = data_frame[col].fillna(data_frame[col].median())
            logger.info(f'Filled {null_count} null values in numeric predictor {col} with median')
    # For categorical columns: fill with mode
    categorical_predictors = data_frame[predictors].select_dtypes(include=['object', 'category']).columns
    for col in categorical_predictors:
        null_count = data_frame[col].isnull().sum()
        if null_count > 0:
            data_frame[col] = data_frame[col].fillna(data_frame[col].mode()[0])
            logger.info(f'Filled {null_count} null values in categorical predictor {col} with mode')

    X = data_frame[predictors].copy()
    y = data_frame[predicted].copy()

    try:
        train_size, valid_size, test_size = splits_sizes

        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        valid_size /= train_size + valid_size
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=valid_size, random_state=seed
        )

        logger.info('Data split completed. Preparing dataset object.')
        dataset = Dataset(train_x, val_x, test_x, train_y, val_y, test_y)

        logger.info('Data preparation complete.')
        return dataset

    except Exception:
        logger.exception('An error occurred during data preparation')
        raise


# def prepare_binary_classfication_tabular_data_from_time_based_datafiles(
#     csv_dirpath: str,
#     predictors: List[str],
#     predicted: str,
#     pos_neg_pair: Optional[Tuple[str, str]] = None,
#     valid_month_count: int = 1,
#     test_month_count: int = 2,
#     all_as_train: bool = False,
# ) -> Dataset:
#     """Prepare the training/validation/test inputs (X) and outputs (y) for binary clasification modeling.

#     Args:
#         csv_dirpath (str): path of the directory of csv files
#         predictors (List[str]): list of predictors column names
#         predicted (str): column name of the predicted outcome
#         pos_neg_pair (Tuple[str,str], optional): groundtruth positive/negative labels. Defaults to None.
#         valid_month_count (int): number of data files for validation.
#         test_month_count (int): number of data files for test.
#         all_as_train (bool): if True, all data files can be used for training. Defaults to False.
#     Returns:
#         Dataset: datassets for binary classification with training/validation/test splits
#     """  
#     def _validate_n_prepare_X_y_dataframes(datafiles):
#         df_X, df_y = pd.DataFrame(), pd.DataFrame()
#         for fpath in datafiles:
#             if not fpath.endswith('.csv'): continue
#             data_frame = pd.read_csv(fpath)
#             if not set(predictors).issubset(data_frame.columns):
#                 missing_cols = set(predictors) - set(data_frame.columns)
#                 raise ValueError(f'Predictor columns missing in data from ({fpath}): {missing_cols}')
#             if predicted not in data_frame.columns:
#                 raise ValueError(f"Predicted column '{predicted}' is missing in data from ({fpath}).")

#             X = data_frame[predictors].copy()
#             y = data_frame[predicted].copy()

#             # Convert labels if pos_neg_pair is provided
#             if pos_neg_pair:
#                 positive, negative = pos_neg_pair
#                 if positive not in y.array or negative not in y.array:
#                     raise ValueError('Specified labels for pos_neg_pair not found in predicted column.')
#                 y = pd.to_numeric(y.replace({positive: '1', negative: '0'}))
#             df_X = pd.concat([df_X, X], axis=0)
#             df_y = pd.concat([df_y, y], axis=0)
#         return df_X, df_y
    
#     train_month_count = len(os.listdir(csv_dirpath)) - valid_month_count - test_month_count
#     if all_as_train:
#         logger.info('All data files will be used for training.')
#         valid_month_count = 0
#         test_month_count = 0
#     else:
#         assert valid_month_count >= 1, f"{valid_month_count} should be at least one."
#         assert test_month_count >= 1, f"{test_month_count} should be at least one."
#         assert train_month_count >= (3 * test_month_count), "assert that train data is at least 3x the test data"

#     try:
#         # Collect file paths and their corresponding dates
#         files_with_dates = [
#             (os.path.join(csv_dirpath, fname), 
#                 datetime.strptime(fname, "extraction_%Y-%m.csv")) for fname in os.listdir(csv_dirpath)
#         ]
#         # Sort files by their datetime (descending: future to past)
#         datafiles = [datafile 
#                      for datafile, _ in sorted(files_with_dates, key=lambda x: x[1], reverse=True)]

#         test_datafiles = datafiles[:test_month_count] 
#         valid_datafiles = datafiles[test_month_count:test_month_count+valid_month_count] 
#         train_datafiles = datafiles[test_month_count+valid_month_count:] 
        
#         train_x, train_y = _validate_n_prepare_X_y_dataframes(train_datafiles)
#         val_x, val_y = _validate_n_prepare_X_y_dataframes(valid_datafiles)
#         test_x, test_y = _validate_n_prepare_X_y_dataframes(test_datafiles)

#         logger.info('Data split completed. Preparing dataset object.')
#         dataset = Dataset(train_x, val_x, test_x, train_y, val_y, test_y)

#         logger.info('Data preparation complete.')
#         return dataset

#     except Exception:
#         logger.exception('An error occurred during data preparation')
#         raise


# def prepare_n_merge_binary_classfication_tabular_data_from_datafiles(
#     csv_dirpath: str,
#     predictors: List[str],
#     predicted: str,
#     pos_neg_pair: Tuple[str, str] | None = None,
#     splits_sizes: Tuple[float] = (0.7, 0.1, 0.2),
#     seed: int = 42,
# ) -> Dataset:
#     """Prepare the training/validation/test inputs (X) and outputs (y) for binary clasification modeling

#     Args:
#     ----
#         csv_dirpath (str): path of the directory of csv files
#         predictors (List[str]): list of predictors column names
#         predicted (str): column name of the predicted outcome
#         pos_neg_pair (Tuple[str,str], optional): groundtruth positive/negative labels. Defaults to None.
#         splits_sizes (List[float], optional): list of relative size portions for training, validation, test data, respectively. Defaults to [0.7,0.1,0.2].
#         seed (int, optional): random seed. Defaults to 42.

#     Returns:
#     -------
#         Dataset: datassets for binary classification with training/validation/test splits
#     """
#     dataset = None
#     for fname in os.listdir(csv_dirpath):
#         if not fname.endswith('.csv'): continue
#         fpath = os.path.join(csv_dirpath, fname)
#         data_frame = pd.read_csv(fpath)
#         if dataset != None:
#             dataset.merge_in(prepare_binary_classfication_tabular_data(data_frame, predictors, predicted, 
#                                                                        pos_neg_pair, splits_sizes, seed))
#         else:
#             dataset = prepare_binary_classfication_tabular_data(data_frame, predictors, predicted, 
#                                                                 pos_neg_pair, splits_sizes, seed)
#     return dataset