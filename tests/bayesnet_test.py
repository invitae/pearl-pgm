from unittest.mock import Mock

import pytest
import torch
import yaml
from pyro import poutine

from pearl.bayesnet import BayesianNetwork, VariableData
from pearl.common import NodeValueType
from pearl.data import BayesianNetworkDataset
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior
from pearl.nodes.continuous import ContinuousNodeWithNormalDistribution

# test Bayesian network
#          +-----+
#          |  a  |
#          +-----+
#         /       \
#        /         \
#       v           v
# +-----+           +-----+
# |  b  |           |  c  |
# +-----+           +-----+
#        \         /
#         \       /
#          v     v
#          +-----+
#          |  d  |
#          +-----+
#
# We assume the following plate structure
# - node b is in ["b_loop"]
# - node c is in ["c_loop"]
# - node d is in ["b_loop", "c_loop"]


@pytest.fixture
def mock_bn():
    test_bn1 = BayesianNetwork("test_bn1", torch.device("cpu"))
    test_bn1.add_plate("b_loop", -2)
    test_bn1.add_plate("c_loop", -1)
    test_bn1.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="a",
        node_parents=[],
        plates=[],
        domain=["False", "True"],
    )
    test_bn1.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="b",
        node_parents=["a"],
        plates=["b_loop"],
        domain=["low", "high"],
    )
    test_bn1.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="c",
        node_parents=["a"],
        plates=["c_loop"],
        domain=["cold", "hot"],
    )
    test_bn1.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="d",
        node_parents=["b", "c"],
        plates=["b_loop", "c_loop"],
        domain=["False", "True"],
    )
    return test_bn1


@pytest.fixture
def mock_bn_tree_plates():
    test_bn1 = BayesianNetwork("test_bn1", torch.device("cpu"))
    test_bn1.add_plate("b_loop", -2)
    test_bn1.add_plate("c_loop", -1)
    test_bn1.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="a",
        node_parents=[],
        plates=[],
        domain=["False", "True"],
    )
    test_bn1.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="b",
        node_parents=["a"],
        plates=["b_loop"],
        domain=["low", "high"],
    )
    test_bn1.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="c",
        node_parents=["a"],
        plates=["c_loop"],
        domain=["cold", "hot"],
    )
    return test_bn1


@pytest.fixture
def mock_plate_dataset():
    variable_dict = {
        "a": VariableData(
            NodeValueType.CATEGORICAL,
            torch.randint(0, 2, (10,)).float(),
            ["False", "True"],
        ),
        "b": VariableData(
            NodeValueType.CATEGORICAL,
            torch.randint(0, 2, (10, 8)).float(),
            ["low", "high"],
        ),
        "c": VariableData(
            NodeValueType.CATEGORICAL,
            torch.randint(0, 2, (10, 6)).float(),
            ["cold", "hot"],
        ),
        "d": VariableData(
            NodeValueType.CATEGORICAL,
            torch.randint(0, 2, (10, 8, 6)).float(),
            ["False", "True"],
        ),
    }
    return BayesianNetworkDataset(variable_dict=variable_dict)


# invalid CPDs (used for testing write to NET file)
bn1_spec = """net
{
\tname = test_bn1;
}
node a
{
\tstates = ("False" "True");
}
node b
{
\tstates = ("low" "high");
}
node c
{
\tstates = ("cold" "hot");
}
node d
{
\tstates = ("False" "True");
}
potential ( a | )
{
\tdata = (0.50000 0.50000);
}
potential ( b | a)
{
\tdata = ((0.50000 0.50000) (0.50000 0.50000));
}
potential ( c | a)
{
\tdata = ((0.50000 0.50000) (0.50000 0.50000));
}
potential ( d | b c)
{
\tdata = (((0.50000 0.50000) (0.50000 0.50000)) ((0.50000 0.50000) (0.50000 0.50000)));
}
"""


def set_instance(bn, dataset):
    for node in bn.dag.nodes:
        node_object = bn.get_node_object(node)
        node_object.set_observed_value(dataset[node][0])


@pytest.mark.parametrize(
    "nodes, predecessors_dict",
    [
        (
            frozenset({"a", "b", "c", "d"}),
            {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]},
        )
    ],
)
def test_add_variable_constructs_dag_correctly(mock_bn, nodes, predecessors_dict):
    # by default model and guide has same graphical model
    assert frozenset(mock_bn.dag.nodes) == nodes
    for node, predecessors in predecessors_dict.items():
        assert list(mock_bn.dag.predecessors(node)) == predecessors


@pytest.mark.parametrize(
    "parent_domain_sizes",
    [
        (
            {
                "a": torch.Size([]),
                "b": torch.Size([2]),
                "c": torch.Size([2]),
                "d": torch.Size([2, 2]),
            }
        )
    ],
)
def test_add_variable_sets_parent_sizes_correctly(mock_bn, parent_domain_sizes):
    for node, size in parent_domain_sizes.items():
        assert mock_bn.dag.nodes[node]["node_object"].parent_domain_sizes == size


def test_set_observed_values(mock_bn, mock_plate_dataset, mark_observed):
    mark_observed(mock_bn, ["b", "c"], False)
    # create dataset with plate sizes [10,8,6]
    mock_bn.set_observed_values(mock_plate_dataset)
    for node, value in [
        ("a", mock_plate_dataset["a"]),
        ("b", None),
        ("c", None),
        ("d", mock_plate_dataset["d"]),
    ]:
        if value is None:
            assert mock_bn.get_node_object(node).observed_value is None
        else:
            assert torch.equal(mock_bn.get_node_object(node).observed_value, value)


@pytest.mark.parametrize(
    "sample_sites", [(["a/cpd", "a", "b/cpd", "b", "c/cpd", "c", "d/cpd", "d"])]
)
def test_model_sample_sites(mock_bn, sample_sites, mock_plate_dataset, mark_observed):
    # this test checks that the named samples encountered by Pyro in a
    # trace of the model are the intended ones.
    mark_observed(mock_bn, [], False)
    set_instance(mock_bn, mock_plate_dataset)
    trace = poutine.trace(mock_bn.model).get_trace(
        plate_sizes={"b_loop": 8, "c_loop": 6}, subsample_sizes={}
    )
    assert all([site in trace.nodes.keys() for site in sample_sites])


@pytest.mark.parametrize(
    "unobserved, sample_sites", [(["b", "c"], ["b/cpd", "c/cpd", "b", "c"])]
)
def test_guide_sample_sites(
    mock_bn, unobserved, sample_sites, mock_plate_dataset, mark_observed
):
    # this test checks that the named samples encountered by Pyro in a
    # trace of the guide are the intended ones.
    mark_observed(mock_bn, unobserved, False)
    set_instance(mock_bn, mock_plate_dataset)
    trace = poutine.trace(mock_bn.guide).get_trace(
        plate_sizes={"b_loop": 8, "c_loop": 6}, subsample_sizes={}
    )
    assert all([site in trace.nodes.keys() for site in sample_sites])


@pytest.mark.parametrize("bn_net_format", [(bn1_spec)])
def test_write_net(tmp_path, mock_bn, bn_net_format, mock_plate_dataset):
    p = tmp_path / "bn.net"
    # evaluate model/guide to populate hyperparams and cpds
    set_instance(mock_bn, mock_plate_dataset)
    mock_bn.model(plate_sizes={"b_loop": 8, "c_loop": 6}, subsample_sizes={})
    mock_bn.guide(plate_sizes={"b_loop": 8, "c_loop": 6}, subsample_sizes={})
    # reset all model hyperparams to 1
    for node_name, node_object in mock_bn.get_node_dict().items():
        node_object.guide_alpha = torch.ones(node_object.guide_alpha.size())

    mock_bn.write_net(p)
    assert p.read_text() == bn_net_format


def test_write_net_assertion_error_with_continuous_node(tmp_path, mock_bn):
    p = tmp_path / "bn.net"
    mock_bn.add_variable(
        node_class=ContinuousNodeWithNormalDistribution,
        node_name="continuous_node",
        node_parents=[],
        plates=[],
    )

    with pytest.raises(AssertionError):
        mock_bn.write_net(p)


@pytest.mark.parametrize("unobserved", [(["b", "c"], [])])
def test_train(mock_bn, unobserved, mock_plate_dataset, mark_observed, optimizer):
    # this is a smoke test for train method.
    # not meant to test convergence of parameter learning
    mark_observed(mock_bn, unobserved, False)
    mock_bn.train(
        dataset=mock_plate_dataset,
        optimizer=optimizer,
        num_steps=100,
        subsample_size=None,
    )


@pytest.mark.parametrize("unobserved, target_variables", [(["b", "c"], ["a"])])
def test_predict(
    mock_bn_tree_plates,
    unobserved,
    target_variables,
    mock_plate_dataset,
    mark_observed,
    optimizer,
):
    mark_observed(mock_bn_tree_plates, unobserved, False)
    mock_bn_tree_plates.train(
        dataset=mock_plate_dataset,
        optimizer=optimizer,
        num_steps=100,
        subsample_size=None,
    )  # this is just to populate the parameters

    # copy dataset for later testing
    dataset_copy = mock_plate_dataset.copy()

    samples, MAP_assignment, assignment_distribution = mock_bn_tree_plates.predict(
        mock_plate_dataset,
        target_variables,
        num_samples=100,
    )

    assert set(samples.keys()) == set(target_variables)
    assert set(assignment_distribution.keys()) == set(target_variables)
    assert set(MAP_assignment.keys()) == set(target_variables)

    assert all(
        [samples[k].shape == (100,) + dataset_copy[k].shape for k in target_variables]
    )

    assert all(
        [MAP_assignment[k].shape == dataset_copy[k].shape for k in target_variables]
    )
    assert all(
        [
            assignment_distribution[k].shape
            == dataset_copy[k].shape + (dataset_copy.discrete_domain_size(k),)
            for k in target_variables
        ]
    )


@pytest.mark.parametrize("unobserved", [(["b", "c"], [])])
def test_train_saves_model_checkpoints(
    mock_bn, unobserved, mock_plate_dataset, mark_observed, optimizer, mocker
):
    # this is a smoke test for train method.
    # not meant to test convergence of parameter learning
    mark_observed(mock_bn, unobserved, False)
    mocker.patch.object(yaml, "dump")
    mock_bn.train(
        dataset=mock_plate_dataset,
        optimizer=optimizer,
        num_steps=100,
        subsample_size=None,
        logdir="/tmp",
    )
    assert yaml.dump.call_count == 10


def test_add_variable_fails_when_node_is_added_before_parents():
    bn = BayesianNetwork("bn", torch.device("cpu"))
    with pytest.raises(KeyError):
        bn.add_variable(
            node_class=CategoricalNodeWithDirichletPrior,
            node_name="child",
            node_parents=["parent"],
            plates=[],
            domain=["yes", "no"],
        )


def test_add_variable_fails_when_node_is_added_before_its_plates():
    bn = BayesianNetwork("bn", torch.device("cpu"))
    with pytest.raises(ValueError):
        bn.add_variable(
            node_class=CategoricalNodeWithDirichletPrior,
            node_name="root",
            node_parents=[],
            plates=["b_loop"],
            domain=["yes", "no"],
        )


def test_add_plates_fails_when_multiple_plates_have_same_name(mock_bn):
    with pytest.raises(ValueError):
        mock_bn.add_plate("b_loop", -2)


def test_add_plates_fails_when_reserved_name_is_used(mock_bn):
    with pytest.raises(ValueError):
        mock_bn.add_plate("data_loop", -2)


def test_add_plates_fails_when_multiple_plates_use_same_dim(mock_bn):
    with pytest.raises(ValueError):
        mock_bn.add_plate("new_plate", -1)


def test_bayesnet_validation_fails_with_cycles(mock_bn):
    # introduce a cycle
    mock_bn.dag.add_edge("d", "a")
    with pytest.raises(ValueError):
        mock_bn.validate(Mock())


def test_bayesnet_validation_fails_if_child_is_outside_parent_plates(
    mock_plate_dataset,
):
    test_bn = BayesianNetwork("test_bn", torch.device("cpu"))
    test_bn.add_plate("plate1", -1)
    test_bn.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="a",
        node_parents=[],
        plates=["plate1"],
        domain=["False", "True"],
    )
    test_bn.add_variable(  # child has no plates
        node_class=CategoricalNodeWithDirichletPrior,
        node_name="b",
        node_parents=["a"],
        plates=[],
        domain=["False", "True"],
    )

    with pytest.raises(ValueError):
        test_bn.validate(mock_plate_dataset)


def test_bayesnet_validation_fails_out_of_domain_values(mock_bn, mock_plate_dataset):
    # Set one of the values for "a" to 2, outside its domain of {0, 1}
    mock_plate_dataset.variable_dict["a"].value[0] = 2

    with pytest.raises(ValueError):
        mock_bn.validate(mock_plate_dataset)
