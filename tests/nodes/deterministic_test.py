import torch

from pearl.common import NodeValueType
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior
from pearl.nodes.deterministic import DeterministicNode


def test_deterministic_node():

    a = CategoricalNodeWithDirichletPrior(
        name="a", domain=["one", "two"], plates=[], parents=[]
    )
    b = CategoricalNodeWithDirichletPrior(
        name="b", domain=["one", "two"], plates=[], parents=[]
    )

    def func(parent_values):
        return parent_values["a"] + parent_values["b"]

    node = DeterministicNode(
        "deterministic",
        func,
        NodeValueType.CATEGORICAL,
        domain=["one", "two"],
        parents=[a, b],
        plates=[],
    )

    a_val = torch.tensor(40.0)
    b_val = torch.tensor(2.0)
    node.sample([], None, a_val, b_val)

    assert node.value == 42
