from deepchecks.tabular import Suite
from deepchecks.tabular.checks import TrainTestPerformance
from deepchecks.tabular.suites import full_suite, model_evaluation

PIPELINE_TEST_SUITE = Suite(
    'Pipeline Test Suite',
    TrainTestPerformance()
    .add_condition_train_test_relative_degradation_less_than(threshold=0.15)
    .add_condition_test_performance_greater_than(0.8),
)

FULL_SUITE = full_suite()

MODEL_EVALUATION_SUITE = model_evaluation()
