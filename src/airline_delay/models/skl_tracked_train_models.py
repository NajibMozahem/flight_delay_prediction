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

"""
This module includes the functions to train and evalure scikit-learn models
for bank marketing ML applications.
"""
import logging
import abc
from typing import Dict, List
from itertools import product
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay
from bank_marketing.data.prep_datasets import Dataset
from bank_marketing.features.skl_build_features import (
    AdvFeatureNames,
    FeatureNames,
    make_advanced_data_transformer,
    make_data_transformer,
)
from bank_marketing.models.skl_train_models import accuracy_evaluation
from bank_marketing.helpers.utils import camel_to_snake, truncated_uuid4
from bank_marketing.mlflow.tracking import Experiment

logger = logging.getLogger('bank_marketing.models.skl_tracked_train_models')

def train_and_evaluate_with_tracking(
    data: Dataset, 
    feature_names: FeatureNames, 
    classifers_list: List[abc.ABCMeta], 
    experiment:Experiment, 
    ds_info:Dict, 
    with_plots:bool=False
) -> List[Dict[str, str | float]]:
    """Train each classifier of the list on the training data then evaluate it on all the splits.

    Args:
        data (Dataset): datasets (training/validation/test)
        feature_names (FeatureNames): features names to be used for classification
        classifers_list (List[abc.ABCMeta]): list of sklearn classifier class
        experiment (Experiment): Experiment settings for MLflow tracking.
    Returns:
        List[Dict[str,str|float]]: A list of dictionaries, where each dictionary contains:
                                    'classifier': The name of the classifier class.
                                    Additional evaluation metrics (e.g., accuracy)
                                    obtained from the accuracy_evaluation function.
    """
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    results = []
    for classifier in classifers_list:
        classifier_name = classifier.__name__
        logger.info('Training and evaluation for classifier: %s', classifier_name)
        classifier_shortname = camel_to_snake(classifier_name)
        with mlflow.start_run(experiment_id=experiment_id, 
                              run_name=f"run_{classifier_shortname}_{truncated_uuid4()}"):
            mlflow.set_tag("sklearn_model", classifier_shortname)
            try:
                model = Pipeline(
                    [
                        ('input_transformer', make_data_transformer(feature_names)),
                        ('classifier', classifier()),
                    ]
                )
                model.fit(data.train_x, data.train_y)
                logger.info('Training completed for classifier: %s', classifier_name)
            except Exception:
                logger.exception('Error training classifier %s', classifier_name)
                continue

            try:
                evaluation_metrics = accuracy_evaluation(model, data)
                # Track accuracy metrics
                mlflow.log_metric("train_accuracy", evaluation_metrics['train'])
                mlflow.log_metric("valid_accuracy", evaluation_metrics['val'])
                mlflow.log_metric("test_accuracy", evaluation_metrics['test'])
                # Generate an example input and a model signature 
                sample = data.train_x.sample(5)
                signature = infer_signature(sample, model.predict(sample))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="classifier",
                    input_example=sample,
                    signature=signature,
                    extra_pip_requirements=["bank-marketing"],
                )
                results.append({'classifier': classifier_name, **evaluation_metrics})
                logger.info(
                    'Evaluation completed for classifier: %s with metrics: %s',
                    classifier_name,
                    evaluation_metrics,
                )
                # Log the ds_info as a YAML file under the run's root artifact directory
                mlflow.log_dict(ds_info, "data.yml")
                if with_plots:
                    # Track ROC curve plots for validation and test sets
                    display = RocCurveDisplay.from_predictions(data.val_y.values,
                                                model.predict_proba(data.val_x)[:,1])
                    mlflow.log_figure(display.figure_, 'plots/ValidRocCurveDisplay.png')
                    display = RocCurveDisplay.from_predictions(data.test_y.values,
                                                model.predict_proba(data.test_x)[:,1])
                    mlflow.log_figure(display.figure_, 'plots/TestRocCurveDisplay.png')
            except Exception:
                logger.exception('Error evaluating classifier %s', classifier_name)

    return results

def advanced_train_and_evaluate_with_tracking(
    data: Dataset,
    feature_names: AdvFeatureNames,
    clustering_algos_list: List[abc.ABCMeta],
    classifers_list: List[abc.ABCMeta],
    experiment:Experiment,
    ds_info:dict,
    with_plots:bool=False
) -> List[Dict[str, str | float]]:
    """Training and evaluation of clustering-based classifier.

    Train each pair of clustering-based feature transformer and classifier of both lists
    on the training data then evaluate it on all the splits.

    Args:
        data (Dataset): datasets (training/validation/test)
        feature_names (AdvFeatureNames): features names to be used for clustering and classification
        clustering_algos_list (List[abc.ABCMeta]): list of sklearn clustering algorithm class
        classifers_list (List[abc.ABCMeta]): list of sklearn classifier class
        experiment (Experiment): Experiment settings
        ds_info (dict): dataset metadata information

    Returns:
        List[Dict[str,str|float]]: accuracy scores w.r.t data splits for all
                                   the pairs of clustering algos and classfiers
    """
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    results = []
    for clustering_algo, classifier in product(clustering_algos_list, classifers_list):
        clustering_name = clustering_algo.__name__
        classifier_name = classifier.__name__
        clustering_shortname = camel_to_snake(clustering_algo.__name__)
        classifier_shortname = camel_to_snake(classifier.__name__)

        logger.info(
            'Training and evaluation for clustering algo %s with classifier %s',
            clustering_name,
            classifier_name,
        )
        with mlflow.start_run(experiment_id=experiment_id, 
                run_name=f"run_{clustering_shortname}_{classifier_shortname}_{truncated_uuid4()}"):
            mlflow.set_tag("sklearn_clustering_model", clustering_shortname)
            mlflow.set_tag("sklearn_classification_model", classifier_shortname)
            try:
                model = Pipeline(
                    [
                        (
                        'input_transformer',
                        make_advanced_data_transformer(feature_names, clustering_algo()),
                        ),
                        ('classifier', classifier()),
                    ]
                )
                model.fit(data.train_x, data.train_y)
                logger.info('Training completed for %s with %s', clustering_name, classifier_name)
            except Exception:
                logger.exception(
                    'Error training pair (%s, %s)',
                    clustering_name,
                    classifier_name,
                )
                continue
            try:
                evaluation_metrics = accuracy_evaluation(model, data)
                # Track metrics
                mlflow.log_metric("train_accuracy", evaluation_metrics['train'])
                mlflow.log_metric("valid_accuracy", evaluation_metrics['val'])
                mlflow.log_metric("test_accuracy", evaluation_metrics['test'])
                # Get Model Signature
                sample = data.train_x.sample(5)
                signature = infer_signature(sample, 
                                            model.predict(sample))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="classifier",
                    input_example=sample,
                    signature=signature,
                    extra_pip_requirements=["bank-marketing"],
                )
                results.append(
                    {
                        'clustering_algo': clustering_name,
                        'classifier': classifier_name,
                        **evaluation_metrics,
                    }
                )
                logger.info(
                    'Evaluation completed for %s with %s: %s',
                    clustering_name,
                    classifier_name,
                    evaluation_metrics,
                )
                # Log the ds_info as a YAML file under the run's root artifact directory
                mlflow.log_dict(ds_info, "data.yml")
                if with_plots:
                    # Track Plots 
                    display = RocCurveDisplay.from_predictions(data.val_y.values,
                                                model.predict_proba(data.val_x)[:,1])
                    mlflow.log_figure(display.figure_, 'plots/ValidRocCurveDisplay.png')
                    display = RocCurveDisplay.from_predictions(data.test_y.values,
                                                model.predict_proba(data.test_x)[:,1])
                    mlflow.log_figure(display.figure_, 'plots/TestRocCurveDisplay.png')
            except Exception:
                logger.exception(
                    'Error evaluating pair (%s, %s)',
                    clustering_name,
                    classifier_name,
                )
    return results

