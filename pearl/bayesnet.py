import functools
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

import networkx as nx
import pyro
import pyro.optim
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO, config_enumerate, infer_discrete

from pearl.common import NodeValueType, Plate, SamplingMode, get_logger
from pearl.data import BayesianNetworkDataset, VariableData
from pearl.hugin import hugin_potential_string
from pearl.nodes.bayesnetnode import Node
from pearl.nodes.categorical import (
    CategoricalNodeWithDirichletPrior,
    GeneralizedLinearNode,
)
from pearl.nodes.continuous import (
    ConditionalLinearGaussianNode,
    ContinuousNodeWithNormalDistribution,
)
from pearl.nodes.deterministic import Exponential, Sum

DATA_LOOP = "data_loop"
PREDICTIVE_SAMPLES = "predictive_samples"
ENCODING_VERSION = "v1.0"  # TODO: versioning system that updates
# automatically (for e.g. using hash of class attributes?)
NODE_CLASS = {
    "CategoricalNodeWithDirichletPrior": CategoricalNodeWithDirichletPrior,
    "ContinuousNodeWithNormalDistribution": ContinuousNodeWithNormalDistribution,
    "GeneralizedLinearNode": GeneralizedLinearNode,
    "ConditionalLinearGaussianNode": ConditionalLinearGaussianNode,
    "Exponential": Exponential,
    "Sum": Sum,
}


class BayesianNetwork:
    """
    Bayesian Network.

    We make the following assumptions about the plates in the
    graphical model:

        - Plates have unique names and dimensions.  The name
          "data_loop" is reserved for use by the library and is used
          for iid samples generated from the BayesianNetwork.  The
          user is required to specify all the plates in the model
          except the "data_loop".

        - If there are 'N' plates (excluding the "data_loop") then
          they should use dimensions -N, -N+1, ..., -1.

        - During training and prediction, the "data_loop" plate is
          added to the model with dimension -N-1.

        - If a node is nested in 'M' plates (excluding the
          "data_loop"), the BayesianNetworkDataset object should
          provide a tensor of M+1 dimensions for that node (with the
          left-most dimension corresponding to the "data_loop" and
          remaining dimensions corresponding to the remaining plates
          of the node).

    We verify that each node is included in all the plates of its
    parents.  This is a general requirement of on plate models and it
    also serves to exclude a class of models for which parallel
    enumeration cannot be performed.  We rely on Pyro implementation
    to detect remaining models for which parallel enumeration is not
    feasible.
    """

    def __init__(self, name: str, device: torch.device) -> None:
        """
        :param name: name of the Bayesian Network
        :param device: cpu/cuda device to use for storing CPD tensors of the nodes
        """

        self.name = name
        self.device = device
        self.dag = nx.DiGraph(name=self.name)
        self.plate_dict = dict()

    def add_plate(self, name: str, dim: int) -> None:
        """
        Add a plate to the Bayesian Nework
        """
        if name in self.plate_dict:
            raise ValueError(f"plate with name {name} already exists")
        if name == DATA_LOOP:
            raise ValueError(f"name {DATA_LOOP} is reserved")
        if dim in [plate.dim for plate in self.plate_dict.values()]:
            raise ValueError(f"dimension {dim} is used by another plate")
        self.plate_dict[name] = Plate(name, dim)

    def get_node_object(self, name: str) -> Optional[Node]:
        return self.dag.nodes[name]["node_object"]

    def get_node_dict(self) -> Mapping[str, Node]:
        return {
            node_name: self.get_node_object(node_name) for node_name in self.dag.nodes
        }

    def _add_node(self, node: Node, parents: List[Node]) -> None:
        assert (
            node.name not in self.dag.nodes
        ), f"node {node.name} already added to the network"
        assert all(
            p.name in self.dag.nodes for p in parents
        ), f"parents of {node.name} not yet added to the network"
        self.dag.add_node(node.name, node_object=node)
        for p in parents:
            self.dag.add_edge(p.name, node.name)

    def add_variable(
        self,
        node_class: Type[Node],
        node_name: str,
        node_parents: List[str],
        plates: List[str],
        **kwargs: dict,
    ) -> None:
        """
        Add a random variable to the Bayesian Network.

        :param node_class: Class from which the node should be instantiated
        :param node_name: name of the random variable
        :param node_parents: names of the parents of the random variable
        :param plates: plates in which the node is embedded
        :param kwargs: arguments to be used in the construction of the node
        """

        if not all(p in self.plate_dict for p in plates):
            raise ValueError(f"plates of {node_name} not yet added to the network")

        parents = [self.get_node_object(p) for p in node_parents]
        node = node_class(
            node_name, plates=plates, parents=parents, device=self.device, **kwargs
        )
        self._add_node(node, parents)

    def sample(
        self,
        *,
        sampling_mode: SamplingMode,
        plate_sizes: Mapping[str, int],
        subsample_sizes: Mapping[str, int],
    ) -> None:
        """
        Sample the nodes of the Bayesian Network.  Consider using one
        of these methods instead.  The sampling_mode parameter is
        frozen for these methods and therefore only the remaining
        arguments may be specified.

            - model: sample from the model distribution

            - guide: sample from the guide distribution

            - MAP_posterior: sample from the MAP of the posterior
              guide distribution

        :param sampling_mode: a SamplingMode enum specifying the sampling mode.
        :param plate_sizes: a dictionary that specifies the size of each plate
        :param subsample_sizes: a dictionary that specifies the subsample sizes of plates
        """

        if not all([plate_name in plate_sizes for plate_name in subsample_sizes]):
            raise ValueError(
                "Plates specified in input subsample_sizes must also appear in input plate_sizes"
            )

        if not all(subsample_sizes[k] <= plate_sizes[k] for k in subsample_sizes):
            raise ValueError("subsample size is greater than plate size")

        # create necessary pyro.plate objects
        for plate_name, pearl_plate in self.plate_dict.items():
            pearl_plate.pyro_plate = pyro.plate(
                plate_name,
                size=plate_sizes[plate_name],
                subsample_size=subsample_sizes.get(plate_name, plate_sizes[plate_name]),
                dim=self.plate_dict[plate_name].dim,
            )

        for node_name in nx.topological_sort(self.dag):
            node_object = self.get_node_object(node_name)
            relevant_plates = [self.plate_dict[p] for p in node_object.plates]
            relevant_plates.sort(key=lambda p: p.dim)  # sort the plates by dimension
            parent_values = tuple(
                self.get_node_object(p).value for p in self.dag.predecessors(node_name)
            )
            node_object.sample(relevant_plates, sampling_mode, *parent_values)

        # clear pyro.plate objects
        for plate_name, pearl_plate in self.plate_dict.items():
            pearl_plate.pyro_plate = None

    # Use functools.partialmethod to freeze the sampling_mode argument
    # of sample method. Since functools.partialmethod appends
    # positional arguments to frozen arguments, we require the use of
    # keyword only arguments for sample and the following methods.
    model = functools.partialmethod(sample, sampling_mode=SamplingMode.MODEL)
    guide = functools.partialmethod(sample, sampling_mode=SamplingMode.GUIDE)

    def MAP_posterior(
        self, plate_sizes: Mapping[str, int], subsample_sizes: Mapping[str, int]
    ) -> None:
        """
        Sample the nodes using the MAP of the posterior guide distribution.
        """
        infer_discrete(
            config_enumerate(self.sample),
            first_available_dim=min(
                (p.dim for p in self.plate_dict.values()), default=0
            )
            - 1,
        )(
            sampling_mode=SamplingMode.MAP_POSTERIOR,
            plate_sizes=plate_sizes,
            subsample_sizes=subsample_sizes,
        )

    def train(
        self,
        dataset: BayesianNetworkDataset,
        optimizer: pyro.optim.PyroOptim,
        num_steps: int,
        subsample_size: int = None,
        logdir: str = None,
    ) -> List[float]:

        """Learn the Conditional Probability Distributions of the
        nodes of the Bayesian Network from the dataset.

        :param dataset: dataset for learning CPDs
        :param optimizer: optimizer object to be used with SVI
        :param num_steps: number of steps of SVI
        :param subsample_size: size of minibatch to be sampled from the dataset
        :param logdir: directory where model checkpoints should be saved
        """
        self.validate(dataset)
        pyro.clear_param_store()

        # add the "data_loop" as the outermost plate
        self._add_outer_plate(DATA_LOOP)
        self.set_observed_values(dataset)

        if subsample_size is None:
            subsample_sizes = {DATA_LOOP: len(dataset)}
        else:
            subsample_sizes = {DATA_LOOP: subsample_size}

        plate_sizes = self._infer_plate_sizes(dataset)

        # Bayesian parameter estimation
        svi = SVI(self.model, self.guide, optim=optimizer, loss=Trace_ELBO())
        losses = []
        logger = get_logger(__name__, logging.INFO)
        if logdir is not None and Path(logdir).exists():
            logdir_path = Path(logdir)
        else:
            if logdir is not None:
                logger.warning(
                    f"directory {logdir} doesn't exist, checkpoints will not saved"
                )
            logdir_path = None
        start_time = time.time()
        for i in range(num_steps):
            loss = svi.step(
                plate_sizes=plate_sizes,
                subsample_sizes=subsample_sizes,
            )
            losses.append(loss)
            if (i + 1) % max(1, (num_steps // 10)) == 0:
                # checkpoint model after every 1/10th of num_steps
                logger.info(f"svi_step: {(i + 1)}")
                logger.info(f"elbo: {loss}")
                if logdir_path is not None:
                    checkpoint_file = str(
                        logdir_path.joinpath(f"{self.name}_{(i + 1)}.yaml")
                    )
                    self._remove_plate(DATA_LOOP)
                    to_yaml(self, checkpoint_file)
                    self._add_outer_plate(DATA_LOOP)
        end_time = time.time()
        logger.info(f"training time: {end_time - start_time}")

        # remove the "data_loop" plate
        self._remove_plate(DATA_LOOP)

        return losses

    def predict(
        self,
        dataset: BayesianNetworkDataset,
        target_variables: List[str],
        num_samples: int = 500,
    ) -> Tuple[dict, dict, dict]:
        """Generate predictions using learned conditional probability
        distributions.  It is required that the ``dataset`` have
        appropriately shaped tensors (with random values) for the
        ``target_variables``.  This is used for inferring plate sizes.

        :param dataset: test dataset for which we need predictions
        :param target_variables: list of variables for which we seek
            predictions
        :param num_samples: number of samples to generate from the
            posterior predictive distribution

        :return: A tuple of three dictionaries.  The dictionaries map
                 from target variables to samples, MAP assignments and
                 distributions
        """
        self.validate(dataset)

        # this might be necessary when the object is loaded from a pickle file
        self.overwrite_param_store()

        # add data_loop plate and infer plate sizes from dataset
        self._add_outer_plate(DATA_LOOP)
        plate_sizes = self._infer_plate_sizes(dataset)

        # add predictive_samples plate
        self._add_outer_plate(PREDICTIVE_SAMPLES)
        plate_sizes[PREDICTIVE_SAMPLES] = num_samples

        # temporarily mark the target variables as unobserved
        old_observed_state = dict()
        for target_variable in target_variables:
            node_object = self.get_node_object(target_variable)
            old_observed_state[target_variable] = node_object.observed
            node_object.observed = False

        # set observed values while expanding dimensions to account for the temporary "predictive_samples" plate
        self.set_observed_values(dataset, dims_to_prepend=(num_samples,))

        self.MAP_posterior(
            plate_sizes=plate_sizes,
            subsample_sizes={},
        )

        # extract predictive samples and determine MAP assignment
        samples = dict()
        target_var_shape = {}
        MAP_assignment, assignment_distribution = dict(), dict()
        for target_var in target_variables:
            sample = self.get_node_object(target_var).value
            plates_dim_size = [
                (self.plate_dict[plate].dim, sample.shape[self.plate_dict[plate].dim])
                for plate in self.get_node_object(target_var).plates
            ]
            target_var_shape[target_var] = tuple(
                [sz for (dim, sz) in sorted(plates_dim_size)]
            )
            samples[target_var] = (
                sample.clone().detach().view(target_var_shape[target_var])
            )

            node_object = self.get_node_object(target_var)
            if node_object.value_type() == NodeValueType.CATEGORICAL:
                counts = torch.zeros(
                    target_var_shape[target_var][
                        1:
                    ]  # need to skip PREDICTIVE_SAMPLES plate
                    + (node_object.domain_size,),
                    device=self.device,
                )
                for i in range(node_object.domain_size):
                    counts[..., i] = torch.eq(samples[target_var], i).sum(dim=0)
                MAP_assignment[target_var] = torch.argmax(counts, dim=-1)
                assignment_distribution[target_var] = counts / num_samples
            else:
                MAP_assignment[target_var] = samples[target_var].mean(dim=0)
                assignment_distribution[target_var] = samples[target_var]

        # remove data_loop and predictive_samples plates
        self._remove_plate(DATA_LOOP)
        self._remove_plate(PREDICTIVE_SAMPLES)

        # undo changes to target variables
        for target_variable in target_variables:
            node_object = self.get_node_object(target_variable)
            node_object.observed = old_observed_state[target_variable]

        return (samples, MAP_assignment, assignment_distribution)

    def set_observed_values(
        self, dataset: BayesianNetworkDataset, dims_to_prepend: Tuple[int] = ()
    ) -> None:
        """Annotate the observed nodes in the graph with observed values. If unobserved then value is set to None.

        :param dataset: the BayesianNetworkDataset object used for annotating the graph with observed values
        :param dims_to_prepend: tuple of dimensions to be prepended to value tensors (used by predict())
        """

        for node_name, node_object in self.get_node_dict().items():
            value = None
            if node_object.observed:
                value = dataset[node_name].clone().detach()

                # prepend dimensions if needed, as in prediction
                value = value.expand(dims_to_prepend + value.shape)

            node_object.set_observed_value(value)

    def generate_dataset(self, plate_sizes: Mapping[str, int]):
        """

        :param plate_sizes: a dictionary that specifies the size of each plate
        :return: dataset object sampled from the model distribution
        """

        if DATA_LOOP not in plate_sizes:
            raise ValueError(
                f"Must specify size for plate {DATA_LOOP} when generating a dataset."
            )

        self._add_outer_plate(DATA_LOOP)

        # temporarily mark all variables as unobserved
        old_observed_state = dict()
        for node_name, node_object in self.get_node_dict().items():
            old_observed_state[node_name] = node_object.observed
            node_object.observed = False

        # sample the model with no observations
        self.model(plate_sizes=plate_sizes, subsample_sizes={})

        variable_dict = {}
        for node_name, node_object in self.get_node_dict().items():
            domain = (
                node_object.domain
                if node_object.value_type() == NodeValueType.CATEGORICAL
                else None
            )
            ordered_plates = list(
                zip(*sorted([(self.plate_dict[p].dim, p) for p in node_object.plates]))
            )[1]
            shape = [plate_sizes[p] for p in ordered_plates]
            value = torch.reshape(node_object.value, shape)

            variable_dict[node_name] = VariableData(
                node_object.value_type(), value, domain
            )

        # remove data_loop and predictive_samples plates
        self._remove_plate(DATA_LOOP)

        # undo changes to target variables
        for node_name, old_state in old_observed_state.items():
            node_object = self.get_node_object(node_name)
            node_object.observed = old_state

        return BayesianNetworkDataset(variable_dict)

    def write_dot(self, dot_file_name: str) -> None:  # pragma: no cover
        """Save the model DAG to a 'dot' file."""
        nx.nx_pydot.write_dot(self.dag, dot_file_name)

    def write_net(self, net_file_name: str) -> None:
        """Save the Bayesian Network in HUGIN NET format."""

        assert all(
            isinstance(n, CategoricalNodeWithDirichletPrior)
            for n in self.get_node_dict().values()
        ), "write_net() not supported for models with non-CATEGORICAL nodes."

        with open(net_file_name, "w") as f:
            # write header
            f.write("net\n")
            f.write("{\n")
            f.write("\tname = {};\n".format(self.name))
            f.write("}\n")

            # write nodes
            for node_name, node_object in self.get_node_dict().items():
                f.write("node {}\n".format(node_name))
                f.write("{\n")
                if node_object.domain is None:
                    domain = range(node_object.domain_size)
                else:
                    domain = node_object.domain
                f.write(
                    "\tstates = ({});\n".format(
                        " ".join('"{}"'.format(w) for w in domain)
                    )
                )
                f.write("}\n")

            # write potentials (i.e., cpds)
            for node_name, node_object in self.get_node_dict().items():
                f.write(
                    "potential ( {} | {})\n".format(
                        node_name, " ".join(self.dag.predecessors(node_name))
                    )
                )
                f.write("{\n")
                _ = node_object.MAP_cpd()
                f.write(
                    "\tdata = {};\n".format(
                        hugin_potential_string(node_object.guide_MAP_cpd)
                    )
                )
                f.write("}\n")

    def overwrite_param_store(self) -> None:
        """Overwrite the parameters in the Pyro param store with own parameters"""
        for node_object in self.get_node_dict().values():
            node_object.overwrite_param_store()

    def validate(self, dataset: BayesianNetworkDataset = None) -> None:
        """
        Check that the graphical model has a valid structure.
        Optionally check that the graphical model and the dataset
        agree on the domain of categorical variables.

            - check that the graph is directed acyclic

            - check that the child nodes don't exist outside of the
              plates of their parents.

            - Given a dataset, check that domain outcomes match
              between the graphical model and the dataset
        """

        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Graph is not a DAG")

        for node in self.dag.nodes:
            node_plates = self.get_node_object(node).plates
            for child in self.dag.successors(node):
                child_plates = self.get_node_object(child).plates
                if not set(node_plates).issubset(child_plates):
                    raise ValueError(
                        f"Plates of node {node} are not included in the plates of its child {child}"
                    )

        if dataset is not None:
            for model_var_name, model_node in self.get_node_dict().items():
                if (
                    model_node.value_type() == NodeValueType.CATEGORICAL
                    and model_var_name in dataset.variable_dict
                ):
                    if model_node.domain != dataset.discrete_domain(model_var_name):
                        raise ValueError(
                            f"Mismatch in domain of categorical variable: "
                            f"{model_node.domain} in model vs. "
                            f"{dataset.discrete_domain(model_var_name)} in dataset."
                        )

                    var_value = dataset[model_var_name]
                    if not torch.all(var_value < model_node.domain_size):
                        raise ValueError(
                            f"Values for node {model_node.name} must be < domain size ({model_node.domain_size}): {var_value}"
                        )

                    if not torch.all(0 <= var_value):
                        raise ValueError(
                            f"Values for node {model_node.name} must be >= 0: {var_value}"
                        )

    def _add_outer_plate(self, name: str) -> None:
        min_dim = min((p.dim for p in self.plate_dict.values()), default=0)
        self.plate_dict[name] = Plate(name, min_dim - 1)
        for node in self.dag.nodes:
            node_object = self.get_node_object(node)
            node_object.plates.append(name)

    def _remove_plate(self, name: str) -> None:
        self.plate_dict.pop(name)
        for node in self.dag.nodes:
            node_object = self.get_node_object(node)
            node_object.plates.remove(name)

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        yaml_encoding = dict()
        yaml_encoding["encodingVersion"] = ENCODING_VERSION
        yaml_encoding["name"] = self.name
        yaml_encoding["device"] = {
            "type": self.device.type,
            "index": self.device.index,
        }
        yaml_encoding["plates"] = {k: v.dim for k, v in self.plate_dict.items()}
        yaml_encoding["nodes"] = dict()
        for node in nx.topological_sort(self.dag):
            node_object = self.get_node_object(node)
            yaml_encoding["nodes"][node] = node_object.to_yaml_encoding()
        return yaml_encoding

    @classmethod
    def from_yaml_encoding(cls, yaml_encoding: Mapping[str, Any]) -> "BayesianNetwork":
        if not yaml_encoding["encodingVersion"] == ENCODING_VERSION:
            raise ValueError(
                f"expected encoding version: {ENCODING_VERSION}, received {yaml_encoding['encodingVersion']}"
            )

        device = torch.device(
            yaml_encoding["device"]["type"], yaml_encoding["device"]["index"]
        )
        model = cls(yaml_encoding["name"], device=device)
        for p, d in yaml_encoding["plates"]:
            model.add_plate(p, d)

        template_dag = nx.DiGraph()
        for n in yaml_encoding["nodes"]:
            template_dag.add_node(n)
            for p in yaml_encoding["nodes"][n]["parents"]:
                template_dag.add_edge(p, n)

        for n in nx.topological_sort(template_dag):
            node_spec = yaml_encoding["nodes"][n]
            node_cls = NODE_CLASS[node_spec["type"]]
            node_object = node_cls.from_yaml_encoding(n, model, node_spec)
            node_parents = [model.get_node_object(p) for p in node_spec["parents"]]
            model._add_node(node_object, node_parents)
        return model

    def _infer_plate_sizes(self, dataset: BayesianNetworkDataset) -> Dict[str, int]:
        """Infer plate sizes from the dataset"""
        plate_sizes = dict()
        for node_name, node_object in self.get_node_dict().items():
            relevant_plates = [self.plate_dict[p] for p in node_object.plates]
            relevant_plates.sort(key=lambda p: p.dim)
            data_dims = dataset[node_name].size()
            for plate, size in zip(relevant_plates, data_dims):
                if plate.name in plate_sizes and plate_sizes[plate.name] != size:
                    raise ValueError(
                        f"dataset doesn't assign a unique size to plate: {plate.name}"
                    )
                else:
                    plate_sizes[plate.name] = size
        return plate_sizes


def to_yaml(bn: BayesianNetwork, file: str) -> None:
    with open(file, "w") as fp:
        yaml.dump(bn.to_yaml_encoding(), fp)


def from_yaml(file: str) -> BayesianNetwork:
    with open(file, "r") as fp:
        yaml_encoding = yaml.safe_load(fp)
        return BayesianNetwork.from_yaml_encoding(yaml_encoding)
