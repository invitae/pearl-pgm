from contextlib import contextmanager

import pyro
import pytest
import torch
from pyro.optim import Adam

from pearl.common import NodeValueType
from pearl.csi import TensorIndexMapping
from pearl.data import BayesianNetworkDataset, VariableData


@pytest.fixture
def does_not_raise():
    def _does_not_raise():
        yield

    return contextmanager(_does_not_raise)


@pytest.fixture(autouse=True)
def clear_param_store():
    pyro.clear_param_store()


@pytest.fixture(autouse=True)
def enable_validation():
    pyro.enable_validation(True)


@pytest.fixture
def plate():
    def _plate(num_samples):
        return pyro.plate("plate", size=num_samples, dim=-1)

    return _plate


@pytest.fixture
def mock_binary_dataset():
    def _mock_binary_dataset(size, varnames):
        variable_dict = {
            varname: VariableData(
                NodeValueType.CATEGORICAL,
                torch.randint(0, 2, (size,)).float(),
                ["False", "True"],
            )
            for varname in varnames
        }
        return BayesianNetworkDataset(variable_dict=variable_dict)

    return _mock_binary_dataset


@pytest.fixture
def optimizer():
    adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
    return Adam(adam_params)


@pytest.fixture
def values():
    def _values(low, high, num_samples):
        return torch.randint(low, high, (num_samples,)).float()

    return _values


@pytest.fixture
def parent_values():
    def _parent_values(size, num_samples):
        return tuple(torch.randint(0, s, (num_samples,)).float() for s in size)

    return _parent_values


@pytest.fixture
def size():
    return torch.Size([2, 3, 4])


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tensor_index_mapping():
    def _tensor_index_mapping(size, device=None):
        return TensorIndexMapping(size, device)

    return _tensor_index_mapping


@pytest.fixture
def mark_observed():
    def _mark_observed(bn, nodes, flag=True):
        for node_name, node_object in bn.get_node_dict().items():
            if node_name in nodes:
                node_object.observed = flag
            else:
                node_object.observed = not flag

    return _mark_observed
