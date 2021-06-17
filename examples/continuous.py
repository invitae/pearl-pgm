from time import time

import torch
from pyro.optim import Adam

from pearl.bayesnet import BayesianNetwork
from pearl.common import NodeValueType
from pearl.data import BayesianNetworkDataset, VariableData
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior
from pearl.nodes.continuous import ContinuousNodeWithNormalDistribution


def mean_from_parent_weights(parent_assignment: torch.tensor):
    # The weights vector is chosen to assign a unique mean to each combination
    # of parents. The dot product of the weights and the parent assignment is
    # taken so the first parent assignment leaves the mean at 0 if unset and
    # increases it to 0.5 if set, the second increases the mean by 0 or 0.25,
    # etc.
    weights = torch.tensor([0.5, 0.25, 0.125])
    assert parent_assignment.shape[0] <= weights.shape[0]

    weights_to_use = weights[0 : parent_assignment.shape[0]]
    return {"loc": torch.dot(parent_assignment, weights_to_use), "scale": 0.01}


# Create a simple dataset with multiple categorical parent nodes and a single
# continuous child node, drawn from one of 2^num_parents different normal
# distributions depending on the parent assignment.
def make_dataset(
    num_samples=1000,
    parent_probs=torch.tensor([0.5, 0.5]),
    child_params_fn=mean_from_parent_weights,
):
    parent_values = torch.stack(
        tuple(
            torch.distributions.Binomial(probs=parent_probs).sample()
            if len(parent_probs) > 0
            else torch.tensor([])
            for _ in range(num_samples)
        )
    )
    child_values = (
        torch.tensor(
            [
                torch.distributions.Normal(
                    **child_params_fn(parent_assignment)
                ).sample()
                for parent_assignment in parent_values
            ]
        )
        .unsqueeze(0)
        .transpose(0, 1)
    )

    # TODO: remove attributes_t and use parent_values and child_values
    # to directly construct the dataset.
    attributes_t = torch.cat((parent_values, child_values), dim=1)
    num_parents = len(parent_probs)

    variable_dict = {
        f"parent_{i}": VariableData(
            NodeValueType.CATEGORICAL,
            attributes_t[:, i],
            ["no", "yes"],
        )
        for i in range(num_parents)
    }
    variable_dict["child"] = VariableData(
        NodeValueType.CONTINUOUS, attributes_t[:, -1], None
    )

    return BayesianNetworkDataset(variable_dict)


# create an 2d attributes tensor for backwards compatibility
def make_attributes_t(dataset):
    return torch.cat([dataset[k].unsqueeze(1) for k in dataset.variable_dict], dim=1)


parent_probs = torch.tensor([0.1, 0.1, 0.5])
num_parents = parent_probs.shape[0]
all_parents = [f"parent_{i}" for i in range(num_parents)]

continuous_model = BayesianNetwork("continuous_model", device=torch.device("cpu"))

for i in range(num_parents):
    continuous_model.add_variable(
        CategoricalNodeWithDirichletPrior,
        f"parent_{i}",
        node_parents=[],
        plates=[],
        domain=["no", "yes"],
    )

continuous_model.add_variable(
    ContinuousNodeWithNormalDistribution, "child", node_parents=all_parents, plates=[]
)

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)
logdir = "/tmp"

train_set_size = 1000
train_dataset = make_dataset(num_samples=train_set_size, parent_probs=parent_probs)

start = time()
continuous_model.train(
    dataset=train_dataset,
    optimizer=optimizer,
    num_steps=10000,
    logdir=logdir,
    subsample_size=None,
)
end = time()

print(f"MAP CPD:\n{continuous_model.get_node_object('child').MAP_cpd()}")
print(f"Training time: {end - start}")

test_set_size = 10
test_dataset = make_dataset(num_samples=test_set_size, parent_probs=parent_probs)

if num_parents > 0:
    # Predict categorical parent values from the continuous child values
    samples, MAP_assignment, assignment_distribution = continuous_model.predict(
        test_dataset,
        all_parents,
    )

    parent_predictions = torch.stack(
        tuple(MAP_assignment[f"parent_{i}"] for i in range(num_parents)), dim=1
    )

    parent_prediction_results = torch.cat(
        (make_attributes_t(test_dataset).float(), parent_predictions.float()), dim=1
    )

    num_predicted = torch.sum(
        parent_prediction_results[:, 0:num_parents]
        == parent_prediction_results[:, num_parents + 1 : num_parents + 1 + num_parents]
    )

    print(f"Parent prediction results:\n{parent_prediction_results}")
    print(
        f"Parent category correctly predicted: {num_predicted} / {test_set_size * num_parents} ({float(num_predicted) / float(test_set_size * num_parents)})"
    )

# Predict continuous child values from the categorical parent values
samples, MAP_assignment, assignment_distribution = continuous_model.predict(
    test_dataset,
    ["child"],
)
# TODO: compute prediction results without constructing attributes_t ?
child_prediction_results = torch.cat(
    (
        make_attributes_t(test_dataset).float(),
        MAP_assignment["child"].unsqueeze(0).transpose(0, 1).float(),
    ),
    dim=1,
)
predict_error = torch.abs(
    child_prediction_results[:, num_parents]
    - child_prediction_results[:, num_parents + 1]
)

mean_predict_error = predict_error.mean()
print(f"Child prediction results:\n{child_prediction_results}")
print(f"Mean prediction error for child: {mean_predict_error}")
