import pytest
import torch

from pearl.bayesnet import BayesianNetworkDataset
from pearl.common import NodeValueType
from pearl.data import VariableData


@pytest.fixture
def dataset():
    variable_dict = {
        "class_variable": VariableData(
            NodeValueType.CATEGORICAL,
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]).float(),
            ["a", "b"],
        ),
        "f1": VariableData(
            NodeValueType.CATEGORICAL,
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1] * 2).float(),
            ["no", "yes"],
        ),
        "f2": VariableData(
            NodeValueType.CATEGORICAL,
            torch.tensor([0, 0, 1, 1] * 4).float(),
            ["no", "yes"],
        ),
        "f3": VariableData(
            NodeValueType.CATEGORICAL, torch.tensor([0, 1] * 8).float(), ["no", "yes"]
        ),
        "f4": VariableData(NodeValueType.CONTINUOUS, torch.rand(16)),
    }
    return BayesianNetworkDataset(variable_dict=variable_dict)
