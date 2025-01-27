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
"""Feature engineering of Bank Marketing.

This module includes the functions and objects to build feature engineering pipelines
for bank marketing ML applications.
"""

import abc
from dataclasses import dataclass
from typing import List

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DateTimeVariablesTransformer(BaseEstimator, TransformerMixin):
    """Creates a new feature indicating the customer has been called before.

    A class used to perform has_been_called_before (new feature) derivation based
    on the value of days_since_last_campaign (existing feature).
    """

    def fit(self, X=None, y=None):
        """Do nothing."""
        return self

    def transform(self, X, y=None):
        """Adds the new feature into the input data."""
        X['has_been_called_before'] = (X.days_since_last_campaign != 999).astype(int)
        X['DEP_HOUR'] = X['DEP_TIME'].fillna(-1).astype(float) // 100
        X['DEP_MINUTE'] = X['DEP_TIME'].fillna(-1).astype(float) % 100

        X['DEP_HOUR'] = X['DEP_HOUR'].where(X['DEP_TIME'].notna(), pd.NA).astype('Int64')
        X['DEP_MINUTE'] = X['DEP_MINUTE'].where(X['DEP_TIME'].notna(), pd.NA).astype('Int64')

        X['FL_DATE'] = pd.to_datetime(X['FL_DATE'])

        X['YEAR'] = X['FL_DATE'].dt.year
        X['MONTH'] = X['FL_DATE'].dt.month
        X['DAY'] = X['FL_DATE'].dt.day
        return X


@dataclass
class FeatureNames:
    """A dataclass used to represent Feature Names for a Basic ML Model.

    Attributes:
    ----------
    numerical : List[str]
        the list of numerical feature names
    categorical : List[str]
        the list of categorical feature names

    Methods:
    -------
    features()
        Returns the list of all features
    """

    numerical: List[str]
    categorical: List[str]

    def features(self) -> List[str]:
        """Returns the list of all features.

        Returns:
            List[str]: list of all features
        """
        return self.numerical + self.categorical


def make_data_transformer(feature_names: FeatureNames) -> Pipeline:
    """Build the scikit-learn based basic data transformer.

    Args:
        feature_names (FeatureNames): names of the numerical & categorical features

    Returns:
        Pipeline: the feature transformer pipeline
    """
    # First create datetime features
    datetime_pipeline = Pipeline([
        ('datetime_features', DateTimeVariablesTransformer()),
    ])
    
    transformer_categorical = Pipeline(
        [
            ('onehot', OneHotEncoder(handle_unknown='error')),
        ]
    )
    transformer_numerical = Pipeline(
        [
            ('scale', StandardScaler()),
        ]
    )
    
    # Update numerical and categorical features to include new datetime features
    numerical_features = feature_names.numerical + ['DEP_HOUR', 'DEP_MINUTE', 'YEAR']
    categorical_features = feature_names.categorical + ['MONTH', 'DAY']
    
    transformer = Pipeline([
        ('datetime', datetime_pipeline),
        ('feature_engineering', ColumnTransformer(
            [
                ('num', transformer_numerical, numerical_features),
                ('cat', transformer_categorical, categorical_features),
            ]
        ))
    ])
    return transformer


@dataclass
class AdvFeatureNames:
    """A dataclass used to represent Feature Names for a Advanced ML Model.

    Attributes:
    ----------
    numerical_clustering : List[str]
        the list of numerical feature names for the clustering algorithm
    categorical_clustering : List[str]
        the list of categorical feature names for the clustering algorithm
    numerical_classifier : List[str]
        the list of numerical feature names for the classifier
    categorical_classifier : List[str]
        the list of categorical feature names for the classifier
    """

    numerical_clustering: List[str]
    categorical_clustering: List[str]
    numerical_classifier: List[str]
    categorical_classifier: List[str]

    def clustering(self) -> List[str]:
        """Returns the list of all features for the classifier."""
        return self.numerical_clustering + self.categorical_clustering

    def classifier(self) -> List[str]:
        """Returns the list of all features for the clustering algorithm."""
        return self.numerical_classifier + self.categorical_classifier


def make_advanced_data_transformer(
    feature_names: AdvFeatureNames, clustering_algo: abc.ABCMeta
) -> Pipeline:
    """Build the scikit-learn based advanced data transformer.

    Args:
        feature_names (FeatureNames): names of the numerical & categorical features
                                      for both clustering & classifier algorithms involved
        clustering_algo: class reference to algorithm implementing the clustering.

    Returns:
        Pipeline: the feature transformer pipeline
    """
    # First create datetime features
    datetime_pipeline = Pipeline([
        ('datetime_features', DateTimeVariablesTransformer()),
    ])
    
    clustering_input_transformer = ColumnTransformer(
        [
            ('scaleCLS', StandardScaler(), feature_names.numerical_clustering),
            ('onehotCLS', OneHotEncoder(sparse_output=True), feature_names.categorical_clustering),
        ],
        remainder='drop',
    )
    clustering_pipeline = Pipeline(
        [
            ('dataCLS', clustering_input_transformer),
            ('algoCLS', clustering_algo),
        ]
    )
    
    transformer_categorical = Pipeline(
        [
            ('onehot', OneHotEncoder(handle_unknown='error')),
        ]
    )
    transformer_numerical = Pipeline(
        [    
            ('scale', StandardScaler()),
        ]
    )
    
    # Update numerical and categorical features to include new datetime features
    numerical_classifier = feature_names.numerical_classifier + ['DEP_HOUR', 'DEP_MINUTE', 'YEAR']
    categorical_classifier = feature_names.categorical_classifier + ['MONTH', 'DAY']
    
    adv_data_transformer = Pipeline([
        ('datetime', datetime_pipeline),
        ('feature_engineering', ColumnTransformer(
            [
                ('clustering', clustering_pipeline, feature_names.clustering()),
                ('num', transformer_numerical, numerical_classifier),
                ('cat', transformer_categorical, categorical_classifier),
            ]
        ))
    ])
    return adv_data_transformer
