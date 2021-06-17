from time import time
from typing import Dict

import torch
from pyro.optim import Adam

from pearl.bayesnet import BayesianNetwork
from pearl.common import NodeValueType, tensor_accuracy
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior
from pearl.nodes.continuous import ContinuousNodeWithNormalDistribution
from pearl.nodes.deterministic import DeterministicNode

# We consider a four node Bayesian Network which has two binary root nodes "A" and "B",
# a categorical deterministic node "D" that can take on [0, 1, 2, 3] based on its parents,
# and a normally distributed child "C" with mean equal to "D".
# there is also a continuous deterministic node "D_cont" that is included as an example
# but does not affect training (until we implement nodes with continuous parents)


def d_func(parent_values: Dict[str, torch.tensor]):
    return 2 * parent_values["A"] + parent_values["B"]


def d_cont_func(parent_values: Dict[str, torch.tensor]):
    return torch.exp(parent_values["A"]) + parent_values["B"]


# create the model
domain_AB = ["false", "true"]
domain_D = ["zero", "one", "two", "three"]
model = BayesianNetwork("model", torch.device("cpu"))
model.add_plate("plate", -1)
model.add_variable(
    CategoricalNodeWithDirichletPrior, "A", [], ["plate"], domain=domain_AB
)
model.add_variable(
    CategoricalNodeWithDirichletPrior, "B", [], ["plate"], domain=domain_AB
)
model.add_variable(
    DeterministicNode,
    "D",
    ["A", "B"],
    ["plate"],
    domain=domain_D,
    node_value_type=NodeValueType.CATEGORICAL,
    func=d_func,
)
model.add_variable(
    DeterministicNode,
    "D_cont",
    ["A", "B"],
    ["plate"],
    node_value_type=NodeValueType.CONTINUOUS,
    func=d_cont_func,
    observed=True,
)

model.add_variable(
    ContinuousNodeWithNormalDistribution,
    "C",
    ["D"],
    ["plate"],
    prior_params={"scale_scale": torch.full((len(domain_D),), 0.1)},
)

# generate synthetic data
plate_sizes = {"data_loop": 10000, "plate": 10}
dataset = model.generate_dataset(plate_sizes)

test_dataset_size = 50
train_dataset, test_dataset = dataset.split(
    (plate_sizes["data_loop"] - test_dataset_size, test_dataset_size)
)

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)

start = time()
loss = model.train(
    dataset=train_dataset,
    optimizer=optimizer,
    num_steps=1000,
    subsample_size=100,
    logdir="/tmp",
)
end = time()
print(f"training time is {end-start}")

start = time()
target_vars = ["A", "B"]
_, MAP_assignment, _ = model.predict(test_dataset, target_vars)
end = time()
print(f"prediction time is {end-start}")

for target_var in target_vars:
    accuracy = tensor_accuracy(test_dataset[target_var], MAP_assignment[target_var])
    print(f"Accuracy in predicting {target_var} given C is {accuracy}")

start = time()
_, MAP_assignment, _ = model.predict(test_dataset, ["C"])
end = time()
print(f"prediction time is {end-start}")

MSE = torch.mean((test_dataset["C"] - MAP_assignment["C"]) ** 2)
print(f"MSE in predicting C given (A, B) is {MSE}")
