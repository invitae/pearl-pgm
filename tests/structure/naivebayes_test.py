import pytest

from pearl.bayesnet import BayesianNetwork
from pearl.structure.naivebayes import categorical_naive_bayes_model


@pytest.mark.parametrize(
    "input_features, expected_features",
    [(None, {"f1", "f2", "f3"}), ({"f1", "f2"}, {"f1", "f2"})],
)
def test_categorical_naive_bayes_construction(
    dataset, input_features, expected_features
):
    nb_model = categorical_naive_bayes_model(
        "nb", dataset, "class_variable", input_features
    )
    assert isinstance(nb_model, BayesianNetwork)
    assert set(nb_model.dag.nodes) == {"class_variable"} | expected_features
    assert set(nb_model.dag.edges) == {("class_variable", f) for f in expected_features}


@pytest.mark.parametrize(
    "class_variable, features",
    [("f4", None), ("class_variable", {"f1", "f2", "f3", "f4"})],
)
def test_categorical_naive_bayes_construction_errors_with_non_categorical_features(
    dataset, class_variable, features
):
    with pytest.raises(ValueError):
        _ = categorical_naive_bayes_model("nb", dataset, class_variable, features)
