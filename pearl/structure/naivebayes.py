from typing import AbstractSet

from pearl.bayesnet import BayesianNetwork
from pearl.data import BayesianNetworkDataset
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior


def categorical_naive_bayes_model(
    name: str,
    dataset: BayesianNetworkDataset,
    class_variable: str,
    features: AbstractSet[str] = None,
) -> BayesianNetwork:
    """
    Create a categorical Naive Bayes model using information from the
    dataset.  This function merely instantiates a BayesianNetwork
    object and doesn't train it on the dataset.

    :param name: name of the BayesianNetwork object to be
        created
    :param dataset: a BayesianNetworkDataset object that provides
        information about the class variable and features.
    :param class_variable: class variable of the model.
    :param features: Set of features to be included in the model.  If
        this argument is None, then all the categorical features in
        the dataset are used.
    """

    if features is None:
        features = set(
            f for f in dataset.variable_dict if dataset.is_categorical(f)
        ) - {class_variable}

    for var in features | {class_variable}:
        if not dataset.is_categorical(var):
            raise ValueError(f"{var} should be a categorical variable in the dataset")

    model = BayesianNetwork(name=name, device=dataset.device)
    model.add_variable(
        node_class=CategoricalNodeWithDirichletPrior,
        node_name=class_variable,
        node_parents=[],
        plates=[],
        domain=dataset.discrete_domain(class_variable),
    )
    for feature in features:
        model.add_variable(
            node_class=CategoricalNodeWithDirichletPrior,
            node_name=feature,
            node_parents=[class_variable],
            plates=[],
            domain=dataset.discrete_domain(feature),
        )
    return model
