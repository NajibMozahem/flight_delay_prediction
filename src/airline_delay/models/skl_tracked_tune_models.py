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
This module includes the functions to tune scikit-learn models
for bank marketing ML applications.
"""
import logging
import abc
from typing import Dict, List
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from hyperopt import fmin, tpe
from sklearn.pipeline import Pipeline
from bank_marketing.data.prep_datasets import Dataset
from bank_marketing.features.skl_build_features import (
    AdvFeatureNames,
    FeatureNames,
    make_advanced_data_transformer,
    make_data_transformer,
)
from bank_marketing.models.skl_train_models import accuracy_evaluation, roc_auc_evaluation
from bank_marketing.helpers.utils import camel_to_snake
from bank_marketing.mlflow.tracking import Experiment

logger = logging.getLogger('bank_marketing.models.skl_tracked_tune_models')
    

def tune_with_tracking(
    data: Dataset, 
    feature_names: FeatureNames, 
    classifer:abc.ABCMeta,
    hparams: Dict,
    max_runs: int, 
    experiment:Experiment, 
    ds_info:Dict
) -> None:
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    hparams_names = list(hparams.keys())
    hparams_space = list(hparams.values())
    fmin(
        fn=build_classic_evaluation_func(data, feature_names, classifer, 
                                         hparams_names, experiment_id, ds_info),
        space=hparams_space,
        algo=tpe.suggest,
        max_evals=max_runs,
    )

def advanced_tune_with_tracking(
    data: Dataset,
    feature_names: AdvFeatureNames,
    clustering_algo:abc.ABCMeta,
    classifier:abc.ABCMeta,
    hparams:Dict,
    max_runs: int,
    experiment:Experiment,
    ds_info:dict
) -> None:
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    parent_run_id = mlflow.active_run().info.run_id
    hparams_names = list(hparams.keys())
    hparams_space = list(hparams.values())
    fmin(
        fn=build_advanced_evaluation_func(data, feature_names, clustering_algo, 
                                          classifier, hparams_names, ds_info, parent_run_id=parent_run_id),
        space=hparams_space,
        algo=tpe.suggest,
        max_evals=max_runs,
    )

def build_classic_evaluation_func(data:Dataset, feature_names:FeatureNames, classifer:abc.ABCMeta, 
                                  hparams_names:List[str], ds_info:dict):
    """
    Create a new evaluation function
    :experiment_id: Experiment id for the training run
    :return: new evaluation function.
    """
    def eval_func(hparams):
        """
        Train sklearn model with given parameters by calling MLflow run.
        :hparam params: Parameters to the train script we optimize over
        :return: The metric value evaluated on the validation data.
        """
        with mlflow.start_run(nested=True):
            classifier_name = camel_to_snake(classifer.__name__)
            mlflow.set_tag("sklearn_model", classifier_name)
            clf_params = {name: value 
                         for name, value in zip(hparams_names,hparams)}
            # Log params
            mlflow.log_params(clf_params)
            model = Pipeline(
                    [
                        ('input_transformer', make_data_transformer(feature_names)),
                        ('classifier', classifer(**clf_params)),
                    ]
                )
            model.fit(data.train_x, data.train_y)
            evaluation_metrics = accuracy_evaluation(model, data)
            roc_auc_metrics = roc_auc_evaluation(model, data)
            # Track accuracy metrics
            mlflow.log_metric("train_accuracy", evaluation_metrics['train'])
            mlflow.log_metric("valid_accuracy", evaluation_metrics['val'])
            mlflow.log_metric("test_accuracy", evaluation_metrics['test'])
            mlflow.log_metrics({
                "train_roc_auc": roc_auc_metrics['train'],
                "valid_roc_auc": roc_auc_metrics['val'],
                "test_roc_auc": roc_auc_metrics['test'],
            })
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
            # Log the ds_info as a YAML file under the run's root artifact directory
            mlflow.log_dict(ds_info, "data.yml")
            return -roc_auc_metrics['val']

    return eval_func


def build_advanced_evaluation_func(data:Dataset, feature_names:AdvFeatureNames, clustering_algo:abc.ABCMeta, 
                                   classifier:abc.ABCMeta, hparams_names:List[str], ds_info:dict, parent_run_id:str):
    """
    Create a new evaluation function
    :experiment_id: Experiment id for the training run
    :return: new evaluation function.
    """
    def eval_func(hparams):
        """
        Train sklearn model with given parameters by calling MLflow run.
        :hparam params: Parameters to the train script we optimize over
        :return: The metric value evaluated on the validation data.
        """
        with mlflow.start_run(parent_run_id=parent_run_id, nested=True):
            clustering_name = camel_to_snake(clustering_algo.__name__)
            classifier_name = camel_to_snake(classifier.__name__)
            mlflow.set_tag("sklearn_clustering_model", clustering_name)
            mlflow.set_tag("sklearn_classification_model", classifier_name)
            cls_params, clf_params = {}, {}
            for name, value in zip(hparams_names, hparams):
                if name in list(clustering_algo().get_params().keys()):
                    cls_params[name] = value
                elif name in list(classifier().get_params().keys()):  
                    clf_params[name] = value
                else:
                    ValueError(f"{name} is not a supported hyperparameter")
            
            mlflow.log_params(clf_params)
            
            model = Pipeline(
                [
                    (
                    'input_transformer',
                    make_advanced_data_transformer(feature_names, clustering_algo(**cls_params)),
                    ),
                    ('classifier', classifier(**clf_params)),
                ]
            )
            mlflow.log_params(model.get_params())

            model.fit(data.train_x, data.train_y)
            evaluation_metrics = accuracy_evaluation(model, data)
            roc_auc_metrics = roc_auc_evaluation(model, data)
            # Track metrics
            mlflow.log_metric("train_accuracy", evaluation_metrics['train'])
            mlflow.log_metric("valid_accuracy", evaluation_metrics['val'])
            mlflow.log_metric("test_accuracy", evaluation_metrics['test'])
            mlflow.log_metrics({
                "train_roc_auc": roc_auc_metrics['train'],
                "valid_roc_auc": roc_auc_metrics['val'],
                "test_roc_auc": roc_auc_metrics['test'],
            })
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
            # Log the ds_info as a YAML file under the run's root artifact directory
            mlflow.log_dict(ds_info, "data.yml")
            return -roc_auc_metrics['val']

    return eval_func