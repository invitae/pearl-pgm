import math
import os

import pytest
import torch

import pearl.bayesnet as bayesnet
import pearl.common as common
import pearl.nodes.categorical as categorical
import pearl.nodes.continuous as continuous
import pearl.nodes.deterministic as deterministic

ABS_TOL = 1e-4


def test_yaml_encoding_for_categorical_with_dirichlet_prior():
    node = categorical.CategoricalNodeWithDirichletPrior(
        name="name",
        domain=["a", "b"],
        plates=["plate1", "plate2"],
        parents=[],
        observed=True,
        prior_params={"alpha": torch.tensor([1.0, 2.0])},
    )
    yaml_encoding = node.to_yaml_encoding()
    assert yaml_encoding.keys() == {
        "type",
        "domain",
        "plates",
        "parents",
        "prior_params",
        "observed",
    }
    assert yaml_encoding["type"] == "CategoricalNodeWithDirichletPrior"
    assert yaml_encoding["domain"] == ["a", "b"]
    assert yaml_encoding["plates"] == ["plate1", "plate2"]
    assert yaml_encoding["parents"] == []
    assert yaml_encoding["observed"]
    assert yaml_encoding["prior_params"].keys() == {"alpha"}
    assert len(yaml_encoding["prior_params"]["alpha"]) == 2
    assert math.isclose(yaml_encoding["prior_params"]["alpha"][0], 1.0, abs_tol=ABS_TOL)
    assert math.isclose(yaml_encoding["prior_params"]["alpha"][1], 2.0, abs_tol=ABS_TOL)


def test_yaml_encoding_for_categorical_node_with_continuous_parents():
    parent1 = categorical.CategoricalNodeWithDirichletPrior(
        name="parent1",
        domain=["a", "b"],
        parents=[],
        plates=[],
    )
    parent2 = continuous.ContinuousNodeWithNormalDistribution(
        name="parent2",
        parents=[],
        plates=[],
    )
    node = categorical.GeneralizedLinearNode(
        name="name",
        domain=["a", "b"],
        plates=["plate1", "plate2"],
        parents=[parent1, parent2],
        observed=True,
        prior_params={
            "bias_mean": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            "bias_scale": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "weights_mean": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
            "weights_scale": torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
        },
    )
    yaml_encoding = node.to_yaml_encoding()
    assert yaml_encoding["type"] == "GeneralizedLinearNode"
    assert yaml_encoding["domain"] == ["a", "b"]
    assert yaml_encoding["plates"] == ["plate1", "plate2"]
    assert yaml_encoding["parents"] == ["parent1", "parent2"]
    assert yaml_encoding["observed"]
    assert set(yaml_encoding["prior_params"].keys()) == {
        "weights_mean",
        "weights_scale",
        "bias_mean",
        "bias_scale",
    }
    assert len(yaml_encoding["prior_params"]["weights_mean"]) == 2
    assert math.isclose(
        yaml_encoding["prior_params"]["weights_mean"][0][0][0], 1.0, abs_tol=ABS_TOL
    )
    assert math.isclose(
        yaml_encoding["prior_params"]["weights_mean"][0][0][1], 2.0, abs_tol=ABS_TOL
    )


def test_yaml_encoding_for_continuous_node_with_continuous_parents():
    parent1 = categorical.CategoricalNodeWithDirichletPrior(
        name="parent1",
        domain=["a", "b"],
        parents=[],
        plates=[],
    )
    parent2 = continuous.ContinuousNodeWithNormalDistribution(
        name="parent2",
        parents=[],
        plates=[],
    )
    node = continuous.ConditionalLinearGaussianNode(
        name="name",
        plates=["plate1", "plate2"],
        parents=[parent1, parent2],
        observed=True,
        prior_params={
            "bias_mean": torch.tensor([0.0, 0.0]),
            "bias_scale": torch.tensor([1.0, 1.0]),
            "weights_mean": torch.tensor([[1.0], [2.0]]),
            "weights_scale": torch.tensor([[5.0], [6.0]]),
            "scale_scale": torch.tensor([1.0, 1.0]),
        },
    )
    yaml_encoding = node.to_yaml_encoding()
    assert yaml_encoding["type"] == "ConditionalLinearGaussianNode"
    assert yaml_encoding["plates"] == ["plate1", "plate2"]
    assert yaml_encoding["parents"] == ["parent1", "parent2"]
    assert yaml_encoding["observed"]
    assert set(yaml_encoding["prior_params"].keys()) == {
        "weights_mean",
        "weights_scale",
        "bias_mean",
        "bias_scale",
        "scale_scale",
    }
    assert len(yaml_encoding["prior_params"]["weights_mean"]) == 2
    assert math.isclose(
        yaml_encoding["prior_params"]["weights_mean"][0][0], 1.0, abs_tol=ABS_TOL
    )
    assert math.isclose(
        yaml_encoding["prior_params"]["weights_mean"][1][0], 2.0, abs_tol=ABS_TOL
    )
    assert math.isclose(
        yaml_encoding["prior_params"]["bias_mean"][0], 0.0, abs_tol=ABS_TOL
    )
    assert math.isclose(
        yaml_encoding["prior_params"]["bias_scale"][0], 1.0, abs_tol=ABS_TOL
    )
    assert math.isclose(
        yaml_encoding["prior_params"]["scale_scale"][0], 1.0, abs_tol=ABS_TOL
    )


def test_yaml_encoding_for_continuous_node_with_normal_distribution():
    node = continuous.ContinuousNodeWithNormalDistribution(
        name="name",
        plates=["plate1", "plate2"],
        parents=[],
        observed=True,
        prior_params={
            "mean_mean": torch.tensor(0.0),
            "mean_scale": torch.tensor(1.0),
            "scale_scale": torch.tensor(1.0),
        },
    )
    yaml_encoding = node.to_yaml_encoding()
    assert yaml_encoding.keys() == {
        "type",
        "plates",
        "parents",
        "prior_params",
        "observed",
    }
    assert yaml_encoding["type"] == "ContinuousNodeWithNormalDistribution"
    assert yaml_encoding["plates"] == ["plate1", "plate2"]
    assert yaml_encoding["parents"] == []
    assert yaml_encoding["observed"]
    assert math.isclose(
        yaml_encoding["prior_params"]["mean_mean"], 0.0, abs_tol=ABS_TOL
    )
    assert math.isclose(
        yaml_encoding["prior_params"]["mean_scale"], 1.0, abs_tol=ABS_TOL
    )
    assert math.isclose(
        yaml_encoding["prior_params"]["scale_scale"], 1.0, abs_tol=ABS_TOL
    )


def test_yaml_encoding_for_exponential_node():
    node = deterministic.Exponential(
        name="name",
        plates=["plate1", "plate2"],
        parents=[],
        observed=False,
    )
    yaml_encoding = node.to_yaml_encoding()
    assert yaml_encoding.keys() == {
        "type",
        "plates",
        "parents",
        "observed",
    }
    assert yaml_encoding["type"] == "Exponential"
    assert yaml_encoding["plates"] == ["plate1", "plate2"]
    assert yaml_encoding["parents"] == []
    assert not yaml_encoding["observed"]


def test_yaml_encoding_for_sum_node():
    node = deterministic.Sum(
        name="name",
        plates=["plate1", "plate2"],
        parents=[],
        observed=False,
    )
    yaml_encoding = node.to_yaml_encoding()
    assert yaml_encoding.keys() == {
        "type",
        "plates",
        "parents",
        "observed",
    }
    assert yaml_encoding["type"] == "Sum"
    assert yaml_encoding["plates"] == ["plate1", "plate2"]
    assert yaml_encoding["parents"] == []
    assert not yaml_encoding["observed"]


def test_yaml_encoding_for_bayesnet(tmp_path):
    bn = bayesnet.BayesianNetwork("bn", torch.device("cpu", 0))
    bn.add_variable(
        categorical.CategoricalNodeWithDirichletPrior,
        "a",
        [],
        [],
        domain=["yes", "no"],
    )
    bn.add_variable(continuous.ContinuousNodeWithNormalDistribution, "b", ["a"], [])
    bn.add_variable(deterministic.Exponential, "c", ["b"], [])
    bn.add_variable(continuous.ContinuousNodeWithNormalDistribution, "d", [], [])
    bn.add_variable(deterministic.Sum, "e", ["c", "d"], [])
    yaml_encoding = bn.to_yaml_encoding()
    assert yaml_encoding.keys() == {
        "encodingVersion",
        "device",
        "name",
        "plates",
        "nodes",
    }
    assert yaml_encoding["encodingVersion"] == bayesnet.ENCODING_VERSION
    assert yaml_encoding["device"] == {"type": "cpu", "index": 0}
    assert yaml_encoding["name"] == "bn"
    assert yaml_encoding["plates"] == {}

    # Since there are separate unit-tests for encoding of nodes we
    # will perform basic checks.
    assert yaml_encoding["nodes"].keys() == {"a", "b", "c", "d", "e"}
    assert yaml_encoding["nodes"]["a"]["parents"] == []
    assert yaml_encoding["nodes"]["a"]["type"] == "CategoricalNodeWithDirichletPrior"
    assert yaml_encoding["nodes"]["b"]["parents"] == ["a"]
    assert yaml_encoding["nodes"]["b"]["type"] == "ContinuousNodeWithNormalDistribution"
    assert yaml_encoding["nodes"]["c"]["parents"] == ["b"]
    assert yaml_encoding["nodes"]["c"]["type"] == "Exponential"
    assert yaml_encoding["nodes"]["d"]["parents"] == []
    assert yaml_encoding["nodes"]["d"]["type"] == "ContinuousNodeWithNormalDistribution"
    assert yaml_encoding["nodes"]["e"]["parents"] == ["c", "d"]
    assert yaml_encoding["nodes"]["e"]["type"] == "Sum"


FIXTURE_DIR = os.path.dirname(os.path.realpath(__file__))
BN_YAML = "bn.yaml"
BN_MINIMAL_YAML = "bn_minimal.yaml"


@pytest.mark.parametrize("yaml_file", [(BN_YAML), (BN_MINIMAL_YAML)])
def test_from_yaml(yaml_file):
    yaml_file_path = os.path.join(FIXTURE_DIR, BN_YAML)
    bn = bayesnet.from_yaml(yaml_file_path)
    assert isinstance(bn, bayesnet.BayesianNetwork)
    assert bn.name == "bn"
    assert common.same_device(bn.device, torch.device("cpu", 0))
    assert set(bn.dag.nodes) == {"a", "b", "c", "d", "e"}
    assert bn.plate_dict == dict()
    assert set(bn.dag.predecessors("a")) == set()
    assert isinstance(
        bn.get_node_object("a"), categorical.CategoricalNodeWithDirichletPrior
    )
    assert set(bn.dag.predecessors("b")) == {"a"}
    assert isinstance(
        bn.get_node_object("b"), continuous.ContinuousNodeWithNormalDistribution
    )
    assert set(bn.dag.predecessors("c")) == {"b"}
    assert isinstance(bn.get_node_object("c"), deterministic.Exponential)
    assert set(bn.dag.predecessors("d")) == set()
    assert isinstance(
        bn.get_node_object("d"), continuous.ContinuousNodeWithNormalDistribution
    )
    assert set(bn.dag.predecessors("e")) == {"c", "d"}
    assert isinstance(bn.get_node_object("e"), deterministic.Sum)
