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
"""Training and evaluation of models for Bank Marketing.

This module includes the functions to train and evalure scikit-learn models
for bank marketing ML applications.
"""

import abc
import logging
from itertools import product
from typing import Dict, List

from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from airline_delay.data.prep_datasets import Dataset
from airline_delay.features.skl_build_features import (
    AdvFeatureNames,
    FeatureNames,
    make_advanced_data_transformer,
    make_data_transformer,
)

logger = logging.getLogger('bank_marketing.models.skl_train_models')


# def roc_auc_evaluation(model: Pipeline, data: Dataset, decimals: int = 3) -> Dict[str, float]:
#     """Compute binary classification AUC scores on data splits by a given model.

#     Args:
#         model (Pipeline): sklearn model pipeline
#         data (Dataset): datasets (training/validation/test)
#         decimals (int, optional): number decimal digits of precision. Defaults to 3.

#     Returns:
#         Dict[str, float]: (keys: splits names, values: AUC scores)
#     """
#     logger.info('Evaluating AUC on training, validation, and testing splits')

#     try:
#         train_auc = roc_auc_score(data.train_y.values, model.predict_proba(data.train_x)[:, 1])
#         logger.info('Training AUC calculated: %.3f', train_auc)
#     except Exception:
#         logger.exception('Error computing training AUC: %s')
#         raise

#     try:
#         val_auc = roc_auc_score(data.val_y.values, model.predict_proba(data.val_x)[:, 1])
#         logger.info('Validation AUC calculated: %.3f', val_auc)
#     except Exception:
#         logger.exception('Error computing validation AUC: %s')
#         raise

#     try:
#         test_auc = roc_auc_score(data.test_y.values, model.predict_proba(data.test_x)[:, 1])
#         logger.info('Testing AUC calculated: %.3f', test_auc)
#     except Exception:
#         logger.exception('Error computing testing AUC: %s')
#         raise

#     auc_scores = {
#         'train': round(train_auc, decimals),
#         'val': round(val_auc, decimals),
#         'test': round(test_auc, decimals),
#     }
#     logger.info('Final AUC scores (rounded to %d decimals): %s', decimals, auc_scores)

#     return auc_scores

def regression_evaluation(model: Pipeline, data: Dataset, decimals: int = 3) -> Dict[str, Dict[str, float]]:
    """Compute regression metrics (MSE, MAE, R2) on data splits by a given model.

    Args:
        model (Pipeline): sklearn model pipeline
        data (Dataset): datasets (training/validation/test)
        decimals (int, optional): number decimal digits of precision. Defaults to 3.

    Returns:
        Dict[str, Dict[str, float]]: (keys: splits names, values: dictionary of regression metrics)
    """
    logger.info('Evaluating regression metrics on training, validation, and testing splits')

    metrics = {}
    
    try:
        # Training metrics
        train_pred = model.predict(data.train_x)
        metrics['train'] = {
            'mse': round(mean_squared_error(data.train_y, train_pred), decimals),
            'mae': round(mean_absolute_error(data.train_y, train_pred), decimals),
            'r2': round(r2_score(data.train_y, train_pred), decimals)
        }
        logger.info('Training metrics calculated: %s', metrics['train'])
    except Exception:
        logger.exception('Error computing training metrics')
        raise

    try:
        # Validation metrics
        val_pred = model.predict(data.val_x)
        metrics['val'] = {
            'mse': round(mean_squared_error(data.val_y, val_pred), decimals),
            'mae': round(mean_absolute_error(data.val_y, val_pred), decimals),
            'r2': round(r2_score(data.val_y, val_pred), decimals)
        }
        logger.info('Validation metrics calculated: %s', metrics['val'])
    except Exception:
        logger.exception('Error computing validation metrics')
        raise

    try:
        # Test metrics
        test_pred = model.predict(data.test_x)
        metrics['test'] = {
            'mse': round(mean_squared_error(data.test_y, test_pred), decimals),
            'mae': round(mean_absolute_error(data.test_y, test_pred), decimals),
            'r2': round(r2_score(data.test_y, test_pred), decimals)
        }
        logger.info('Test metrics calculated: %s', metrics['test'])
    except Exception:
        logger.exception('Error computing test metrics')
        raise

    logger.info('Final metrics (rounded to %d decimals): %s', decimals, metrics)
    return metrics


def train_and_evaluate(
    data: Dataset, feature_names: FeatureNames, regressors_list: List[abc.ABCMeta]
) -> List[Dict[str, str | Dict[str, float]]]:
    """Train each regressor of the list on the training data then evaluate it on all the splits.

    Args:
        data (Dataset): datasets (training/validation/test)
        feature_names (FeatureNames): features names to be used for regression
        regressors_list (List[abc.ABCMeta]): list of sklearn regressor classes

    Returns:
        List[Dict[str,str|Dict[str,float]]]: A list of dictionaries, where each dictionary contains:
                                    'model_name': The name of the regressor class.
                                    Additional evaluation metrics (MSE, MAE, R2)
                                    obtained from the regression_evaluation function.
    """
    results = []
    for regressor in regressors_list:
        model_name = regressor.__name__
        logger.info('Training and evaluation for regressor: %s', model_name)

        try:
            model = Pipeline(
                [
                    ('input_transformer', make_data_transformer(feature_names)),
                    ('regressor', regressor()),
                ]
            )
            model.fit(data.train_x, data.train_y)
            logger.info('Training completed for regressor: %s', model_name)
        except Exception:
            logger.exception('Error training regressor %s', model_name)
            continue

        try:
            evaluation_metrics = regression_evaluation(model, data)
            results.append({'model_name': model_name, **evaluation_metrics})
            logger.info(
                'Evaluation completed for regressor: %s with metrics: %s',
                model_name,
                evaluation_metrics,
            )
        except Exception:
            logger.exception('Error evaluating regressor %s', model_name)

    return results


# def advanced_train_and_evaluate(
#     data: Dataset,
#     feature_names: AdvFeatureNames,
#     clustering_algos_list: List[abc.ABCMeta],
#     classifers_list: List[abc.ABCMeta],
# ) -> List[Dict[str, str | float]]:
#     """Training and evaluation of clustering-based classifier.

#     Train each pair of clustering-based feature transformer and classifier of both lists
#     on the training data then evaluate it on all the splits.

#     Args:
#         data (Dataset): datasets (training/validation/test)
#         feature_names (AdvFeatureNames): features names to be used for clustering and classification
#         clustering_algos_list (List[abc.ABCMeta]): list of sklearn clustering algorithm class
#         classifers_list (List[abc.ABCMeta]): list of sklearn classifier class

#     Returns:
#         List[Dict[str,str|float]]: accuracy scores w.r.t data splits for all
#                                    the pairs of clustering algos and classfiers
#     """
#     results = []
#     for clustering_algo, classifier in product(clustering_algos_list, classifers_list):
#         clustering_name = clustering_algo.__name__
#         classifier_name = classifier.__name__

#         logger.info(
#             'Training and evaluation for clustering algo %s with classifier %s',
#             clustering_name,
#             classifier_name,
#         )

#         try:
#             model = Pipeline(
#                 [
#                     (
#                         'input_transformer',
#                         make_advanced_data_transformer(feature_names, clustering_algo),
#                     ),
#                     ('classifier', classifier()),
#                 ]
#             )
#             model.fit(data.train_x, data.train_y)
#             logger.info('Training completed for %s with %s', clustering_name, classifier_name)
#         except Exception:
#             logger.exception(
#                 'Error training pair (%s, %s)',
#                 clustering_name,
#                 classifier_name,
#             )
#             continue

#         try:
#             evaluation_metrics = accuracy_evaluation(model, data)
#             results.append(
#                 {
#                     'clustering_algo': clustering_name,
#                     'classifier': classifier_name,
#                     **evaluation_metrics,
#                 }
#             )
#             logger.info(
#                 'Evaluation completed for %s with %s: %s',
#                 clustering_name,
#                 classifier_name,
#                 evaluation_metrics,
#             )
#         except Exception:
#             logger.exception(
#                 'Error evaluating pair (%s, %s)',
#                 clustering_name,
#                 classifier_name,
#             )

#     return results


def print_regression_scoring(results: List[Dict[str, Dict[str, float]]] | Dict[str, Dict[str, float]]) -> None:
    """Print the regression scoring results as pretty tables.

    Args:
        results (List[Dict[str, Dict[str, float]]] | Dict[str, Dict[str, float]]): 
            Either a single model's results or a list of results from multiple models.
            Each result contains metrics (MSE, MAE, R2) for train/val/test splits.
    """
    if isinstance(results, dict):
        # Single model results
        tab = PrettyTable()
        tab.field_names = ['Split', 'MSE', 'MAE', 'R2']
        for split in ['train', 'val', 'test']:
            if split in results:
                metrics = results[split]
                tab.add_row([
                    split,
                    metrics['mse'],
                    metrics['mae'],
                    metrics['r2']
                ])
    elif isinstance(results, list):
        # Multiple models results
        tab = PrettyTable()
        tab.field_names = ['Model', 'Split', 'MSE', 'MAE', 'R2']
        for result in results:
            model_name = result.get('model_name', 'Unknown')
            for split in ['train', 'val', 'test']:
                if split in result:
                    metrics = result[split]
                    tab.add_row([
                        model_name,
                        split,
                        metrics['mse'],
                        metrics['mae'],
                        metrics['r2']
                    ])
    print(tab)
