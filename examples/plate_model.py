from time import time

import torch
from pyro.optim import Adam

from pearl.bayesnet import BayesianNetwork
from pearl.common import tensor_accuracy
from pearl.nodes.categorical import CategoricalNodeWithDirichletPrior
from pearl.nodes.continuous import ContinuousNodeWithNormalDistribution

# We consider a four node Bayesian Network which has the graph
# structure shown in the unit tests and has a mixture distribution
# similar to the one given in continuous.py

# The root variable is named "A" and it has two children "I" and
# "N". While "I" tracks "A" with high probability, "N" negates "A"
# with high probability. Finally there is node "M" which is normally
# distributed and whose mean is governed by "I" and "N". We assume the
# nodes "I", "N" are in a plate named "plate", "M" is embedded in
# "plate" and a "nested_plate".

model = BayesianNetwork("plate_model", torch.device("cpu"))
model.add_plate("plate", -2)
model.add_plate("nested_plate", -1)
model.add_variable(CategoricalNodeWithDirichletPrior, "A", [], [], domain=["no", "yes"])
model.add_variable(
    CategoricalNodeWithDirichletPrior, "I", ["A"], ["plate"], domain=["no", "yes"]
)
model.add_variable(
    CategoricalNodeWithDirichletPrior, "N", ["A"], ["plate"], domain=["no", "yes"]
)
model.add_variable(
    ContinuousNodeWithNormalDistribution,
    "M",
    ["I", "N"],
    ["plate", "nested_plate"],
    prior_params={"scale_scale": torch.full((2, 2), 0.01)},
)

# generate synthetic data assuming that size of data_loop=1000 and size of nested_plate=10
num_instances = 1000
num_plates = 10
num_nested_plates = 5

dataset = model.generate_dataset(
    {"data_loop": num_instances, "plate": num_plates, "nested_plate": num_nested_plates}
)

test_dataset_size = 50
train_dataset, test_dataset = dataset.split(
    (num_instances - test_dataset_size, test_dataset_size)
)

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)

start = time()
model.train(
    dataset=train_dataset,
    optimizer=optimizer,
    num_steps=1000,
    subsample_size=None,
    logdir="/tmp",
)
end = time()
print(f"training time is {end-start}")

start = time()
target_vars = ["A", "I", "N"]
_, MAP_assignment, _ = model.predict(
    test_dataset,
    target_vars,
)
end = time()
print(f"prediction time is {end-start}")

for target_var in target_vars:
    accuracy = tensor_accuracy(test_dataset[target_var], MAP_assignment[target_var])
    print(f"Accuracy in predicting {target_var} is {accuracy}")
