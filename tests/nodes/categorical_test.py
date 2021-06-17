import pyro
import pytest
import torch
from pyro.ops.indexing import vindex

from pearl.common import Plate, SamplingMode
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior


@pytest.mark.parametrize(
    "node_class, parent_domain_sizes, kwargs, parent_values, expected_exception, expected_shape",
    [
        (
            CategoricalNodeWithDirichletPrior,
            [2, 3],
            {"plates": [], "domain": ["low", "medium", "high"]},
            (),
            AssertionError,
            None,
        ),
        (
            CategoricalNodeWithDirichletPrior,
            [2, 3],
            {"plates": [], "domain": ["low", "medium", "high"]},
            (torch.tensor([0.0, 0.0]),),
            AssertionError,
            None,
        ),
        (
            CategoricalNodeWithDirichletPrior,
            [2, 3],
            {"plates": [], "domain": ["low", "medium", "high"]},
            (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])),
            None,
            torch.Size([3, 3]),
        ),
    ],
)
def test_index_cpd_result_shape(
    node_class,
    parent_domain_sizes,
    parent_values,
    kwargs,
    expected_exception,
    does_not_raise,
    expected_shape,
    make_parents_with_domain_sizes,
):
    parents = make_parents_with_domain_sizes(parent_domain_sizes)

    node = node_class("node", parents=parents, **kwargs)
    cpd_tensor = torch.rand(
        torch.Size([p.domain_size for p in parents]) + (len(kwargs["domain"]),)
    )
    node.model_cpd = cpd_tensor
    if not expected_exception:
        with does_not_raise():
            cpd = node._index_cpd(node.model_cpd, *parent_values)
            assert cpd.shape == expected_shape
    else:
        with pytest.raises(expected_exception):
            _ = node._index_cpd(node.model_cpd, *parent_values)


@pytest.mark.parametrize(
    "parent_domain_sizes, parent_values",
    [
        ([], ()),
        (
            [2, 3, 4],
            (
                torch.tensor([0.0, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 1.0]),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "node_class, kwargs",
    [
        (
            CategoricalNodeWithDirichletPrior,
            {"plates": [], "domain": ["low", "medium", "high"]},
        )
    ],
)
def test_index_cpd_result_values(
    node_class,
    parent_domain_sizes,
    kwargs,
    parent_values,
    make_parents_with_domain_sizes,
):
    parents = make_parents_with_domain_sizes(parent_domain_sizes)
    node = node_class("node", parents=parents, **kwargs)
    node.sample_model_cpd()
    selected_cpds = node._index_cpd(node.model_cpd, *parent_values)
    if len(parents) > 0:
        t_parent_values = tuple(p.long() for p in parent_values)
        expected_cpds = vindex(node.model_cpd, t_parent_values)
    else:
        expected_cpds = node.model_cpd
    assert torch.equal(selected_cpds, expected_cpds)


# TODO: add an _index_cpd test which uses context specific independence


@pytest.mark.parametrize("batch_size", [(1), (100)])
@pytest.mark.parametrize(
    "node_class, kwargs",
    [
        (
            CategoricalNodeWithDirichletPrior,
            {
                "plates": [],
                "parents": [],
                "domain": ["low", "medium", "high"],
                "observed": False,
            },
        )
    ],
)
def test_sample_model_unobserved_nodes(node_class, kwargs, batch_size, mocker, plate):
    node = node_class("node", **kwargs)
    node.set_observed_value(torch.zeros(batch_size))
    mocker.spy(node, "sample_model_cpd")
    mocker.spy(pyro, "sample")
    data_loop = Plate("data_loop", -1, plate(batch_size))
    node.sample([data_loop], SamplingMode.MODEL)
    node.sample_model_cpd.assert_called_once()
    # check arguments of last pyro.sample
    pyro.sample.assert_called_with(mocker.ANY, mocker.ANY, obs=None)
