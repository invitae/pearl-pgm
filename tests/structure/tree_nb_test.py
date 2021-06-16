import math

import networkx as nx
import pytest
import torch

import pearl
from pearl.bayesnet import BayesianNetwork
from pearl.data import VariableData
from pearl.structure.tree_augmented_nb import (
    augmenting_tree,
    categorical_tree_augmented_naive_bayes,
    conditional_mutual_information,
    empirical_probability,
    log_empirical_conditional_probability,
)


def test_categorical_tree_augmented_naive_bayes_construction(dataset):
    """
    Example based unit-test for categorical_tree_augmented_naive_bayes()
    """
    model = categorical_tree_augmented_naive_bayes(
        "tree_nb", dataset, "class_variable", "f1"
    )
    assert isinstance(model, BayesianNetwork)
    assert set(model.dag.nodes) == {"class_variable", "f1", "f2", "f3"}


def test_augmenting_tree(dataset):
    """
    Example based unit-test for augmenting_tree()
    """
    projected_dataset = dataset.project(["class_variable", "f1", "f2", "f3"])
    tree = augmenting_tree(projected_dataset, "class_variable", "f1")
    assert nx.is_directed(tree)
    assert nx.is_tree(tree)
    assert set(tree.nodes) == {"f1", "f2", "f3"}
    assert tree.in_degree["f1"] == 0


def test_conditional_mutual_information_of_independent_variables(dataset):
    """
    Example based unit-test for conditional mutual information
    """
    projected_dataset = dataset.project(["class_variable", "f1", "f2", "f3"])
    # the dataset contains uniformly distributed variables and hence
    # the conditional mutual information is zero
    torch.testing.assert_allclose(
        conditional_mutual_information(
            projected_dataset, "f1", "f2", "class_variable", 16
        ),
        0,
    )


def test_conditional_mutual_information_of_correlated_variables(dataset):
    """
    Example based unit-test for conditional mutual information
    """
    # make f1 and f2 highly correlated
    # increase the size of the dataset to minimize the effect of additive smoothing
    repeat = 10000
    variable_dict = dict()
    for var in ["class_variable", "f1", "f2", "f3"]:
        variable_dict[var] = VariableData(
            dataset.variable_dict[var].value_type,
            dataset.variable_dict[var].value.repeat(repeat),
            dataset.variable_dict[var].discrete_domain,
        )
    projected_dataset = pearl.data.BayesianNetworkDataset(variable_dict)
    # make f1 and f2 observations identical
    projected_dataset["f2"] = projected_dataset["f1"]
    torch.testing.assert_allclose(
        conditional_mutual_information(
            projected_dataset, "f1", "f2", "class_variable", 16
        ),
        1,
    )


def test_log_empirical_conditional_probability(dataset):
    """
    Example based unit-test for log_empirical_conditional_probability()
    """

    projected_dataset = dataset.project(["class_variable", "f1", "f2", "f3"])
    # since the dataset contains uniformly distributed variables, the
    # conditional probability of any pair of assignments is 0.5
    torch.testing.assert_allclose(
        log_empirical_conditional_probability(
            projected_dataset, {"f1": "yes"}, {"f2": "no"}, 16
        ),
        math.log2(0.5),
    )


def test_empirical_probability(dataset):
    """
    Example based unit-test for empirical_probability()
    """
    projected_dataset = dataset.project(["class_variable", "f1", "f2", "f3"])
    # since the dataset contains uniformly distributed variables, the
    # probability of a full assignment is 1 / len(dataset)
    torch.testing.assert_allclose(
        empirical_probability(
            projected_dataset,
            {"class_variable": "a", "f1": "yes", "f2": "no", "f3": "yes"},
            16,
        ),
        1 / len(projected_dataset),
    )


@pytest.mark.parametrize(
    "assignment, expected_count, expected_smoothing_factor",
    [
        ({"class_variable": "a", "f1": "yes", "f2": "yes", "f3": "yes"}, 1, 1 / 16),
        ({"class_variable": "a", "f1": "yes", "f2": "yes"}, 2, 1 / 8),
        ({"class_variable": "a", "f1": "yes"}, 4, 1 / 4),
        ({"class_variable": "a"}, 8, 1 / 2),
    ],
)
def test_empirical_probability_smoothing(
    dataset, assignment, expected_count, expected_smoothing_factor
):
    """
    Example based unit-test for additive smoothing done by empirical_probability()
    """
    projected_dataset = dataset.project(["class_variable", "f1", "f2", "f3"])
    torch.testing.assert_allclose(
        empirical_probability(projected_dataset, assignment, 16),
        (expected_count + expected_smoothing_factor) / (16 + 1),
    )


def test_augmenting_tree_root_should_be_in_features(dataset):
    with pytest.raises(ValueError):
        _ = categorical_tree_augmented_naive_bayes(
            "model", dataset, "class_variable", "f1", {"f2", "f3"}
        )


def test_features_should_be_categorical(dataset):
    with pytest.raises(ValueError):
        _ = categorical_tree_augmented_naive_bayes(
            "model", dataset, "class_variable", "f1", {"f1", "f2", "f3", "f4"}
        )
