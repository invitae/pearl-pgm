import pytest
import torch

from pearl.common import NodeValueType
from pearl.data import BayesianNetworkDataset, VariableData


@pytest.fixture
def model_2_dataset():
    N = 1000
    a = torch.distributions.Categorical(probs=torch.tensor([0.3, 0.7])).sample((N,))
    variable_dict = {
        "a": VariableData(
            NodeValueType.CATEGORICAL,
            a.float(),
            ["low", "high"],
        )
    }
    return BayesianNetworkDataset(variable_dict)


@pytest.fixture
def model_1_dataset():
    N = 1000
    a = (
        torch.distributions.Categorical(
            probs=torch.softmax(torch.tensor([0.0, 1.0]), dim=-1)
        )
        .sample((N,))
        .float()
    )
    b = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0)).sample((N,))
    mask = torch.eq(a, 0.0)
    c = torch.empty((N,))
    c[mask] = torch.distributions.Normal(torch.tensor(-1.0), torch.tensor(0.1)).sample(
        (N,)
    )[mask]
    c[~mask] = torch.distributions.Normal(torch.tensor(1.0), torch.tensor(0.1)).sample(
        (N,)
    )[~mask]
    d = torch.distributions.Normal(b * 1.0 + 2, torch.tensor(0.1)).sample()
    e = (
        torch.distributions.Categorical(
            probs=torch.softmax(
                torch.stack(
                    [
                        0.0 + c * 1.0 + d * 2.0,
                        1.0 + c * 4.0 + d * 6.0,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )
        )
        .sample()
        .float()
    )
    variable_dict = {
        "a": VariableData(
            NodeValueType.CATEGORICAL,
            a,
            discrete_domain=["low", "high"],
        ),
        "b": VariableData(
            NodeValueType.CONTINUOUS,
            b,
        ),
        "c": VariableData(
            NodeValueType.CONTINUOUS,
            c,
        ),
        "d": VariableData(
            NodeValueType.CONTINUOUS,
            d,
        ),
        "e": VariableData(
            NodeValueType.CATEGORICAL,
            e,
            ["yes", "no"],
        ),
    }
    return BayesianNetworkDataset(variable_dict)
