import math
from functools import reduce
from itertools import combinations, product
from typing import AbstractSet, Mapping

import networkx as nx
import torch
from tqdm import tqdm

from pearl.bayesnet import BayesianNetwork
from pearl.data import BayesianNetworkDataset
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior


def categorical_tree_augmented_naive_bayes(
    name: str,
    dataset: BayesianNetworkDataset,
    class_variable: str,
    augmenting_tree_root: str,
    features: AbstractSet[str] = None,
) -> BayesianNetwork:
    """
    Create a categorical tree augmented Naive Bayes model using
    information from the dataset.  This function merely instantiates a
    BayesianNetwork object and doesn't train it on the dataset.

    :param name: name of the BayesianNetwork object to be created
    :param dataset: a BayesianNetworkDataset object that provides
        information for creating the tree augmented naive bayes model.
    :param class_variable: class variable of the model.
    :param augmenting_tree_root: feature to be used as the root of the
        augmenting tree.  Should be included in 'features' parameter
        if it is not None.
    :param features: Set of features to be included in the model.  If
        this argument is None, then all the categorical features in
        the dataset are used.
    """

    if features is None:
        features = set(
            f for f in dataset.variable_dict if dataset.is_categorical(f)
        ) - {class_variable}

    if augmenting_tree_root not in features:
        raise ValueError(
            f"{augmenting_tree_root} should be part of the feature variables"
        )

    for var in features | {class_variable, augmenting_tree_root}:
        if not dataset.is_categorical(var):
            raise ValueError(f"{var} should be a categorical variable in the dataset")

    projected_dataset = dataset.project(
        list(features | {class_variable, augmenting_tree_root})
    )

    model = BayesianNetwork(name, dataset.device)
    model.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name=class_variable,
        node_parents=[],
        plates=[],
        domain=dataset.discrete_domain(class_variable),
    )
    model.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name=augmenting_tree_root,
        node_parents=[class_variable],
        plates=[],
        domain=dataset.discrete_domain(augmenting_tree_root),
    )

    for pred, succ in augmenting_tree(
        projected_dataset, class_variable, augmenting_tree_root
    ).edges:
        model.add_variable(
            node_class=CategoricalNodeWithDirichletPrior,
            node_name=succ,
            node_parents=[class_variable, pred],
            plates=[],
            domain=dataset.discrete_domain(succ),
        )
    return model


def augmenting_tree(
    dataset: BayesianNetworkDataset, class_variable: str, augmenting_tree_root: str
) -> nx.Graph:
    """
    Construct the directed tree whose edges are used for augmenting
    the Naive Bayes model.  Its constructed as follows

        - create a complete graph whose edges are weighted by
          conditional mutual information

        - construct the maximum spanning tree of the above graph

        - make the maximum spanning tree directed by using
          'augmenting_tree_root' as the root of BFS

    :param dataset: BayesianNetworkDataset object used for
        constructing maximum spanning tree.
    :param class_variable: class variable
    :param augmenting_tree_root: variable to be used as the root of
        the augmenting tree
    """

    g = nx.Graph()
    features = set(dataset.variable_dict.keys()) - {class_variable}
    num_total_instantiations = reduce(
        lambda x, y: x * y,
        [dataset.discrete_domain_size(v) for v in dataset.variable_dict],
    )
    for node1, node2 in tqdm(combinations(features, 2)):
        cmi = conditional_mutual_information(
            dataset, node1, node2, class_variable, num_total_instantiations
        )
        g.add_edge(node1, node2, weight=cmi)
    max_spanning_tree = nx.maximum_spanning_tree(g)
    return nx.bfs_tree(max_spanning_tree, augmenting_tree_root)


def conditional_mutual_information(
    dataset: BayesianNetworkDataset,
    x: str,
    y: str,
    z: str,
    num_total_instantiations: int,
) -> float:
    """
    Compute conditional mutual information using the dataset.

    :param dataset: dataset used for computing conditional mutual information
    :param x: random variable (see below formula)
    :param y: random variable (see below formula)
    :param z: random variable (see below formula)
    :param num_total_instantiations: total number of possible
        instantiations of random variables in the dataset

    I_p(X, Y | Z) = sum_{x, y, z} P(x, y, z) * log(P(x, y | z) / P(x | z) * P(y | z))
    """

    x_domain = dataset.discrete_domain(x)
    y_domain = dataset.discrete_domain(y)
    z_domain = dataset.discrete_domain(z)

    cmi = 0.0  # conditional mutual information
    for xval, yval, zval in product(x_domain, y_domain, z_domain):
        term1 = empirical_probability(
            dataset, {x: xval, y: yval, z: zval}, num_total_instantiations
        )
        term2 = log_empirical_conditional_probability(
            dataset, {x: xval, y: yval}, {z: zval}, num_total_instantiations
        )
        term3 = log_empirical_conditional_probability(
            dataset, {x: xval}, {z: zval}, num_total_instantiations
        )
        term4 = log_empirical_conditional_probability(
            dataset, {y: yval}, {z: zval}, num_total_instantiations
        )
        cmi += term1 * (term2 - term3 - term4)
    return cmi


def log_empirical_conditional_probability(
    dataset: BayesianNetworkDataset,
    assignment1: Mapping[str, str],
    assignment2: Mapping[str, str],
    num_total_instantiations: int,
) -> float:
    """
    Compute log P(assignment1 | assignment2)

    :param dataset: dataset used for computing conditional probability
    :param assignment1: variable assignment
    :param assignment2: variable assignment
    :param num_total_instantiations: total number of possible
        instantiations of random variables in the dataset
    """
    term1 = empirical_probability(
        dataset, {**assignment1, **assignment2}, num_total_instantiations
    )
    term2 = empirical_probability(dataset, assignment2, num_total_instantiations)
    return math.log2(term1) - math.log2(term2)


def empirical_probability(
    dataset: BayesianNetworkDataset,
    assignment: Mapping[str, str],
    num_total_instantiations: int,
) -> float:
    """
    Compute the empirical probability of a random variable assignment
    according to the dataset.  Additive smoothing is used to avoid
    zero probabilities.

    :param dataset: dataset used for computing empirical probability
    :param assignment: random variable assignment for which empirical
        probability is to be computed
    :param num_total_instantiations: total number of possible
        instantiations of random variables in the dataset
    """
    smoothing_multiplicative_factor = num_total_instantiations
    for var in assignment.keys():
        smoothing_multiplicative_factor /= dataset.discrete_domain_size(var)
    count = (
        int(torch.sum(dataset.select(assignment)))
        + 1 / num_total_instantiations * smoothing_multiplicative_factor
    )
    return count / (len(dataset) + 1)
