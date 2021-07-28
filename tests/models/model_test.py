import math
import pathlib

import pytest
import torch

from pearl.bayesnet import from_yaml
from tests.markers import needs_cuda, slow

ABS_TOL = 1e-4
YAMLS_PATH = pathlib.Path(__file__).parent.joinpath("yaml_files")


@pytest.fixture
def model_1():
    yaml_file = YAMLS_PATH.joinpath("model1.yaml").resolve()
    model = from_yaml(str(yaml_file))
    return model


def train_test_split(dataset):
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_data, test_data = dataset.split((n_train, n_test))
    return (train_data, test_data)


@slow
def test_model_1(model_1, model_1_dataset, optimizer):
    model = model_1
    train_data, test_data = train_test_split(model_1_dataset)
    model.train(
        dataset=train_data,
        optimizer=optimizer,
        num_steps=10000,
    )
    _, map_assignments, _ = model.predict(
        dataset=test_data,
        target_variables=["e"],
    )
    acc = float(torch.eq(test_data["e"], map_assignments["e"]).sum()) / len(test_data)
    assert math.isclose(acc, 1.0, abs_tol=0.1)


@slow
@needs_cuda
def test_model_1_cuda(model_1, model_1_dataset, optimizer):
    cuda_device = torch.device("cuda", 0)

    model = model_1.to(cuda_device)
    cuda_dataset = model_1_dataset.to(cuda_device)
    train_data, test_data = train_test_split(cuda_dataset)

    model.train(
        dataset=train_data,
        optimizer=optimizer,
        num_steps=10000,
    )
    _, map_assignments, _ = model.predict(
        dataset=test_data,
        target_variables=["e"],
    )
    acc = float(torch.eq(test_data["e"], map_assignments["e"]).sum()) / len(test_data)
    assert math.isclose(acc, 1.0, abs_tol=0.1)
