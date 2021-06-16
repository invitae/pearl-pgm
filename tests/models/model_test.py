import math
import os
import pathlib

import pytest
import torch

from pearl.bayesnet import from_yaml

ABS_TOL = 1e-4
YAMLS_PATH = pathlib.Path(__file__).parent.joinpath("yaml_files")


slow = pytest.mark.skipif(
    not os.getenv("RUN_SLOW_TESTS", False),
    reason="enable slow running tests by setting 'RUN_SLOW_TESTS' environment variable",
)


@slow
def test_model_2(model_2_dataset, optimizer):
    yaml_file = YAMLS_PATH.joinpath("model2.yaml").resolve()
    model = from_yaml(str(yaml_file))
    n_train = int(0.8 * len(model_2_dataset))
    n_test = len(model_2_dataset) - n_train
    train_data, test_data = model_2_dataset.split((n_train, n_test))
    model.train(
        dataset=train_data,
        optimizer=optimizer,
        num_steps=5000,
        plate_sizes=dict(),
        subsample_sizes=dict(),
        logging_disabled=True,
    )
    _, map_assignments, _ = model.predict(
        test_data,
        ["a"],
        plate_sizes=dict(),
    )
    acc = float(torch.eq(test_data["a"], map_assignments["a"]).sum()) / len(test_data)
    assert math.isclose(acc, 1.0, abs_tol=ABS_TOL)


@slow
def test_model_1(model_1_dataset, optimizer):
    yaml_file = YAMLS_PATH.joinpath("model1.yaml").resolve()
    model = from_yaml(str(yaml_file))
    n_train = int(0.8 * len(model_1_dataset))
    n_test = len(model_1_dataset) - n_train
    train_data, test_data = model_1_dataset.split((n_train, n_test))
    model.train(
        dataset=train_data,
        optimizer=optimizer,
        num_steps=10000,
        plate_sizes=dict(),
        subsample_sizes=dict(),
        logging_disabled=True,
    )
    _, map_assignments, _ = model.predict(
        test_data,
        ["e"],
        plate_sizes=dict(),
    )
    acc = float(torch.eq(test_data["e"], map_assignments["e"]).sum()) / len(test_data)
    assert math.isclose(acc, 1.0, abs_tol=0.1)
