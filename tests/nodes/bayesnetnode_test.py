import pyro
import pytest
import torch

from pearl.common import Plate, SamplingMode
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior
from pearl.nodes.continuous import ContinuousNodeWithNormalDistribution


@pytest.mark.parametrize(
    "node_class, kwargs",
    [
        (
            CategoricalNodeWithDirichletPrior,
            {"plates": [], "parents": [], "domain": ["low", "medium", "high"]},
        )
    ],
)
def test_nodes_without_parents_dont_have_csi(node_class, kwargs):
    node = node_class("node", **kwargs)
    assert node.model_csi is None
    assert node.guide_csi is None


@pytest.mark.parametrize("event_dims", [(()), ((2,)), ((2, 3))])
@pytest.mark.parametrize(
    "node_class, kwargs",
    [
        (
            CategoricalNodeWithDirichletPrior,
            {"plates": [], "domain": ["low", "medium", "high"]},
        )
    ],
)
def test_add_csi_rule(node_class, kwargs, event_dims, make_parents_with_domain_sizes):
    node = node_class(
        "node", parents=make_parents_with_domain_sizes([2, 3, 4]), **kwargs
    )
    assert node.model_csi is not None
    node.model_add_csi_rule({0: 0, 1: 1})
    node.guide_add_csi_rule({0: 0, 1: 1})
    # (0, 1, 2) should be mapped to (0, 1, 0)
    cpd = torch.rand((2, 3, 4) + event_dims)
    cpd_with_csi = node.model_csi.map(cpd)
    assert torch.eq(cpd_with_csi[0, 1, 2], cpd_with_csi[0, 1, 0]).all()
    cpd_with_csi = node.guide_csi.map(cpd)
    assert torch.eq(cpd_with_csi[0, 1, 2], cpd_with_csi[0, 1, 0]).all()


@pytest.mark.parametrize(
    "node_class, parent_domain_sizes, kwargs, expected_hyperparams",
    [
        (
            CategoricalNodeWithDirichletPrior,
            [2, 3, 4],
            {"plates": [], "domain": ["low", "medium", "high"]},
            {"guide_alpha": torch.ones(2, 3, 4, 3)},
        ),
        (
            ContinuousNodeWithNormalDistribution,
            [2, 3, 4],
            {
                "plates": [],
            },
            {
                "guide_mean_mean": torch.zeros(2, 3, 4),
                "guide_mean_scale": torch.ones(2, 3, 4),
                "guide_scale": torch.ones(2, 3, 4),
            },
        ),
    ],
)
def test_sample_guide_cpd_sets_guide_hyperparams(
    node_class,
    parent_domain_sizes,
    kwargs,
    expected_hyperparams,
    make_parents_with_domain_sizes,
):
    parents = make_parents_with_domain_sizes(parent_domain_sizes)
    node = node_class("node", parents=parents, **kwargs)
    node.sample_guide_cpd()
    for attr_name, expected_value in expected_hyperparams.items():
        torch.testing.assert_allclose(getattr(node, attr_name), expected_value)


@pytest.mark.parametrize("parent_domain_sizes", [[], [2, 3, 4]])
@pytest.mark.parametrize(
    "node_class, kwargs, event_shape",
    [
        (
            CategoricalNodeWithDirichletPrior,
            {"plates": [], "domain": ["low", "medium", "high"]},
            (3,),
        ),
        (ContinuousNodeWithNormalDistribution, {"plates": []}, (2,)),
    ],
)
def test_sample_cpd_initializes_cpd_to_the_right_shape(
    node_class, parent_domain_sizes, kwargs, event_shape, make_parents_with_domain_sizes
):
    parents = make_parents_with_domain_sizes(parent_domain_sizes)
    node = node_class("node", parents=parents, **kwargs)
    node.sample_model_cpd()
    node.sample_guide_cpd()
    assert (
        node.model_cpd.shape
        == torch.Size([p.domain_size for p in parents]) + event_shape
    )
    assert (
        node.guide_cpd.shape
        == torch.Size([p.domain_size for p in parents]) + event_shape
    )


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


@pytest.mark.parametrize(
    "batch_size, values, expected_exception",
    [
        (1, torch.zeros(1), None),
        (100, torch.zeros(100), None),
        (100, torch.zeros(50), IndexError),
    ],
)
@pytest.mark.parametrize(
    "node_class, kwargs",
    [
        (
            CategoricalNodeWithDirichletPrior,
            {
                "plates": [],
                "parents": [],
                "domain": ["low", "medium", "high"],
                "observed": True,
            },
        )
    ],
)
def test_sample_model_observed_nodes(
    node_class, kwargs, batch_size, values, expected_exception, plate, does_not_raise
):
    node = node_class("node", **kwargs)
    node.set_observed_value(values)
    data_loop = Plate("data_loop", -1, plate(batch_size))
    if not expected_exception:
        with does_not_raise():
            node.sample([data_loop], SamplingMode.MODEL)
            assert len(node.value) == batch_size
            assert torch.equal(values, node.value)
    else:
        with pytest.raises(expected_exception):
            node.sample([data_loop], SamplingMode.MODEL)


@pytest.mark.parametrize("batch_size", [(1), (100)])
@pytest.mark.parametrize(
    "node_class, domain",
    [(CategoricalNodeWithDirichletPrior, ["low", "medium", "high"])],
)
@pytest.mark.parametrize("observed, expected_exception", [(False, None), (True, None)])
def test_sample_guide(
    node_class,
    batch_size,
    plate,
    domain,
    does_not_raise,
    observed,
    expected_exception,
):
    node = node_class("node", plates=[], parents=[], domain=domain, observed=observed)
    node.set_observed_value(torch.zeros(batch_size))
    data_loop = Plate("data_loop", -1, plate(batch_size))
    if not expected_exception:
        with does_not_raise():
            node.sample([data_loop], SamplingMode.GUIDE)
            assert len(node.value) == batch_size
    else:
        with pytest.raises(expected_exception):
            node.sample([data_loop], SamplingMode.GUIDE)


# TODO: combine tests for model and guide sampling. Also include sampling from posterior


@pytest.mark.parametrize(
    "node_class, parent_domain_sizes, kwargs, expected_prior_params",
    [
        (
            CategoricalNodeWithDirichletPrior,
            [],
            {"plates": [], "domain": ["low", "medium", "high"]},
            {"alpha": torch.ones(3)},
        ),
        (
            CategoricalNodeWithDirichletPrior,
            [2, 3, 4],
            {"plates": [], "domain": ["low", "medium", "high"]},
            {"alpha": torch.ones(2, 3, 4, 3)},
        ),
        (
            CategoricalNodeWithDirichletPrior,
            [2, 3, 4],
            {
                "plates": [],
                "domain": ["low", "medium", "high"],
                "prior_params": {"alpha": torch.full((2, 3, 4, 3), 5.0)},
            },
            {"alpha": torch.full((2, 3, 4, 3), 5.0, dtype=torch.float)},
        ),
        (
            ContinuousNodeWithNormalDistribution,
            [],
            {
                "plates": [],
            },
            {
                "mean_mean": torch.zeros([]),
                "mean_scale": torch.ones([]),
                "scale_scale": torch.ones([]),
            },
        ),
        (
            ContinuousNodeWithNormalDistribution,
            [2, 3, 4],
            {
                "plates": [],
            },
            {
                "mean_mean": torch.zeros([2, 3, 4]),
                "mean_scale": torch.ones([2, 3, 4]),
                "scale_scale": torch.ones([2, 3, 4]),
            },
        ),
        (
            ContinuousNodeWithNormalDistribution,
            [2, 3, 4],
            {
                "plates": [],
                "prior_params": {
                    "mean_mean": torch.full([2, 3, 4], 0.5, dtype=torch.float),
                    "mean_scale": torch.full([2, 3, 4], 10, dtype=torch.float),
                },
            },
            {
                "mean_mean": torch.full([2, 3, 4], 0.5, dtype=torch.float),
                "mean_scale": torch.full([2, 3, 4], 10, dtype=torch.float),
                "scale_scale": torch.ones([2, 3, 4]),
            },
        ),
    ],
)
def test_informative_priors(
    node_class,
    parent_domain_sizes,
    kwargs,
    expected_prior_params,
    make_parents_with_domain_sizes,
):
    node = node_class(
        "node", parents=make_parents_with_domain_sizes(parent_domain_sizes), **kwargs
    )
    for expected_param_name, expected_param_value in expected_prior_params.items():
        torch.testing.assert_allclose(
            node.prior_params[expected_param_name], expected_param_value
        )


@pytest.mark.parametrize(
    "kwargs, prior_params, expected_exception",
    [
        (
            {"plates": [], "parents": [], "domain": ["low", "medium", "high"]},
            {"alpha": torch.full((3,), 5.0, dtype=torch.float)},
            None,
        ),
        (
            {"plates": [], "parents": [], "domain": ["low", "medium", "high"]},
            {"alpha": torch.full((3,), -5.0, dtype=torch.float)},
            AssertionError,
        ),
    ],
)
def test_alpha_limits(
    kwargs,
    prior_params,
    expected_exception,
    does_not_raise,
):
    if expected_exception is None:
        with does_not_raise():
            _ = CategoricalNodeWithDirichletPrior(
                "node",
                **kwargs,
                prior_params=prior_params,
            )
    else:
        with pytest.raises(expected_exception):
            _ = CategoricalNodeWithDirichletPrior(
                "node",
                **kwargs,
                prior_params=prior_params,
            )
