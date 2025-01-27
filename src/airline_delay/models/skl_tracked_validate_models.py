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

import os
import warnings
import tempfile

warnings.filterwarnings("ignore")
from typing import Dict
import logging
import mlflow
from sklearn.pipeline import Pipeline
from bank_marketing.data.prep_datasets import Dataset
from deepchecks import SuiteResult
from deepchecks.tabular import Dataset as DeepChecksDataset
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import TrainTestPerformance
from bank_marketing.mlflow.registry import load_model, get_registered_model_metadata
from bank_marketing.mlflow.registry import tag_model
from bank_marketing.mlflow.registry import transition_model_to_staging
from bank_marketing.helpers.utils import create_temporary_dir_if_not_exists
from bank_marketing.helpers.utils import clean_temporary_dir
from bank_marketing.helpers.utils import get_matched_keys

logger = logging.getLogger('bank_marketing.models.skl_tracked_validate_models')

def validate_model(
        dataset: Dataset, 
        data_card: dict,
        model: Pipeline,
        suite: Suite
):
    categorical_cols = [feature for key in get_matched_keys(data_card, "*cat*_features*")
                        for feature in data_card[key]]
    ds_train = DeepChecksDataset(dataset.train_x, label=dataset.train_y, 
                                 cat_features=categorical_cols)
    ds_test =  DeepChecksDataset(dataset.val_x,  label=dataset.val_y, 
                                 cat_features=categorical_cols)

    result = suite.run(model=model,
                       train_dataset=ds_train,
                       test_dataset=ds_test)
    
    logger.info(f" {len(result.get_passed_checks())} of Model tests are passed.")
    logger.info(f" {len(result.get_not_passed_checks())} of Model tests are failed.")
    logger.info(f" {len(result.get_not_ran_checks())} of Model tests are not runned.")

    return result


def save_validation_report_in_mlflow(result: SuiteResult, run_id: str, mlflow_client: mlflow.client.MlflowClient | None = None, json_report_kwargs: dict | None = None):
    # Note it is important to use a distinct temporary directory for each run so we can support parallel runs
    # (if not the cleanup of the temporary directory would remove the report of the other run)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_fpath = lambda fpath: os.path.join(tmp_dir, fpath)
        with open(tmp_fpath("deepchecks_report.json"), 'w') as f:
            f.write(result.to_json(**(json_report_kwargs or {})))
        result.save_as_html(tmp_fpath("deepchecks_report.html"))

        mlflow_client = mlflow_client or mlflow.client.MlflowClient()
        mlflow_client.log_artifacts(run_id, tmp_dir, artifact_path="tests")


def validate_with_tracking(
        dataset: Dataset, 
        data_card:Dict,
        registry_uri:str,
        model_name:str,
        model_version:str
):
    categorical_cols = [feature for key in get_matched_keys(data_card, "*cat*_features*")
                        for feature in data_card[key]]
    ds_train = DeepChecksDataset(dataset.train_x, label=dataset.train_y, 
                                 cat_features=categorical_cols)
    ds_test =  DeepChecksDataset(dataset.val_x,  label=dataset.val_y, 
                                 cat_features=categorical_cols)
    model:Pipeline = load_model(registry_uri, model_name, model_version)

    custom_suite = Suite('Pipeline Test Suite',
                            # TODO: add customized model checks
                            TrainTestPerformance()\
                            .add_condition_train_test_relative_degradation_less_than(
                                threshold=0.15
                            )\
                            .add_condition_test_performance_greater_than(0.8)
                        )
        
    result = custom_suite.run(model=model, 
                                train_dataset=ds_train, 
                                test_dataset=ds_test)

    save_validation_report_in_mlflow(result, model.run_id)

    logger.info(f" {len(result.get_passed_checks())} of Model tests are passed.")
    logger.info(f" {len(result.get_not_passed_checks())} of Model tests are failed.")
    logger.info(f" {len(result.get_not_ran_checks())} of Model tests are not runned.")

    if result.passed(fail_if_check_not_run=True, fail_if_warning=True):
        print("The Model validation succeeds")
        tag_model(registry_uri, model_name, model_version, {"Model Validation Tests": "PASSED"})
        transition_model_to_staging(registry_uri, model_name, model_version)
    else:
        print("The Model validation fails")
        tag_model(registry_uri, model_name, model_version, {"Model Validation Tests": "FAILED"})
