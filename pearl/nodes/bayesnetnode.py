from __future__ import annotations

from contextlib import ExitStack
from typing import Any, List, Mapping, Optional, Sequence

import pyro
import pyro.distributions as distributions
import torch
from pyro.ops.indexing import Vindex

from pearl.common import NodeValueType, Plate, SamplingMode, same_device
from pearl.csi import TensorIndexMapping


class Node:
    """Node in a Bayesian Network"""

    def __init__(
        self,
        name: str,
        plates: List[str],
        parents: Sequence[Node],
        device: torch.device = torch.device("cpu"),
        prior_params: Mapping[str, torch.Tensor] = None,
        observed: bool = True,
    ) -> None:
        """
        :param name: name of the node (i.e., random variable)
        :param plates: list of plates in which the node is embedded
        :param parents: list of parent nodes
        :param device: cpu/cuda device to use for storing tensors
        :param prior_params: a mapping from param name to prior values.  See implementations for specifics of what
        prior params are accepted and their interpretation.
        :param observed: whether the node is observed or hidden
        """
        self.name = name
        self.device = device
        self.parents = parents

        self.prior_params = dict(
            self.get_default_prior_params(), **(prior_params or {})
        )

        self.validate_prior_params()
        self.observed = observed
        self.value = None  # sampled value; can have enumeration + batch dimensions
        self.observed_value = None  # observed value; can have batch dimensions
        self.plates = [] if plates is None else list(plates)

    def set_observed_value(
        self, value: Optional[torch.Tensor]
    ) -> None:  # pragma: no cover
        """Set the value of the node."""
        if value is not None:
            assert same_device(value.device, self.device)

        self.observed_value = value

    def sample_model_cpd(self) -> torch.Tensor:  # pragma: no cover
        """Sample the model CPD from its prior."""
        raise NotImplementedError

    def sample_guide_cpd(self) -> torch.Tensor:  # pragma: no cover
        """Sample the guide CPD from its prior."""
        raise NotImplementedError

    def MAP_cpd(self) -> torch.Tensor:  # pragma: no cover
        """Record the MAP of the posterior on the CPD of the node and return it."""
        raise NotImplementedError

    def sample(
        self,
        relevant_plates: List[Plate],
        sampling_mode: SamplingMode,
        *parent_values: torch.Tensor,
    ) -> None:
        """
        First obtain the node-specific distribution given any parent values.
        If this node is hidden then sample from the distribution.
        If this node is observed but we are currently guide mode then skip any sampling.
        If this node is observed and we are not currently in guide mode then create a sample
         site for the distributuion from with observations set.
        """

        sample_dist = self.get_sample_dist(sampling_mode, *parent_values)

        with ExitStack() as stack:
            idxs = tuple(
                stack.enter_context(plate.pyro_plate) for plate in relevant_plates
            )
            if not self.observed:
                # unobserved nodes are always sampled
                self.value = pyro.sample(self.name, sample_dist, obs=None).float()
            else:
                obs = self.subsample_observations(relevant_plates, idxs)
                # observed nodes are sampled except in guide
                if sampling_mode == SamplingMode.GUIDE:
                    self.value = obs
                else:
                    self.value = pyro.sample(self.name, sample_dist, obs=obs).float()

    def get_sample_dist(
        self, sampling_mode: SamplingMode, *parent_values: torch.Tensor
    ) -> distributions.Distribution:
        """Given any parent values, return the Pyro distribution from which the node's values should be sampled."""
        raise NotImplementedError

    def subsample_observations(
        self, relevant_plates: List[Plate], idxs: Sequence[torch.tensor]
    ):
        """Broadcast the 1d indices returned by plates so that data has the right batch shape

        Let -m be the minimum dimension (most leftmost/negative) of the relevant plates for this node. Then the
        pyro.sample site for this node is expecting observations that have dimensions from -m to -1. Ultimately,
        the left-most plate during training and prediction, data_loop and predictive_samples, respectively,
        are added to every node. Therefore all pyro.sample sites in pearl will expect observations with number of
        dimensions equal to the total number of plates.

        Observations in the user-created dataset for each node only contain dimensions for the plates relevant to
        that node (data_loop plus any user-defined plates). This method reshapes the self.observed_value tensor to
        fill out those missing dimensions (along with actually subsampling). As an example, a variable that lies on
        plates with dim index -4 and -2 will be have self.observed_value as a 2-dimensional tensor with size [x,
        y]. This method will reshape it to a tensor with size [x, 1, y, 1].

        This is done by reshaping each of the indices coming from the plate contexts in relevant_plates. Each of
        those indices idx for a plate at dimension -d need to get singleton dimensions tacked on from [-d+1, ...,
        -1] so that idx has shape [plate_size, 1, ..., 1], with d dimensions. Once the indices are broadcaseted,
        Vindex takes care of everything else, including dealing with enumerated dimensions.
        """

        # if dim=-N add abs(N) - 1 singleton dimensions to the right.
        dims = tuple(plate.dim for plate in relevant_plates)
        broadcasted_idxs = tuple(
            idx.view((-1,) + (1,) * (abs(dim) - 1)) for idx, dim in zip(idxs, dims)
        )
        obs = Vindex(self.observed_value)[broadcasted_idxs]

        return obs

    def overwrite_param_store(self) -> None:  # pragma: no cover
        """Overwrite node parameters in the Pyro param store with current node attributes."""
        raise NotImplementedError

    def value_type(self) -> NodeValueType:  # pragma: no cover
        """
        :return value type of node, whether categorical or continuous.
        """
        raise NotImplementedError

    def validate_prior_params(self) -> None:
        """Ensure that the prior params that are passed in are valid. In general, each class's method should call
        that of its parent class as well.
        """
        for tensor in self.prior_params.values():
            assert same_device(tensor.device, self.device)

    def get_default_prior_params(self) -> Mapping[str, torch.Tensor]:
        """
        :return str --> torch.Tensor dictionary of configured default params.
        """
        return dict()

    def dist(self, tensor) -> distributions.Distribution:  # pragma: no cover
        """
        :return the distribution used for this class.
        """
        raise NotImplementedError

    def __eq__(self, other) -> bool:  # pragma: no cover
        """
        Test whether the node is equal to the 'other' node.
        """
        raise NotImplementedError

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        yaml_encoding = dict()
        yaml_encoding["type"] = type(self).__name__
        yaml_encoding["plates"] = list(self.plates)
        yaml_encoding["parents"] = [p.name for p in self.parents]
        yaml_encoding["observed"] = self.observed
        return yaml_encoding

    @classmethod
    def process_yaml_spec(
        cls, bayesnet, yaml_encoding: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        kwargs = dict()
        kwargs["plates"] = yaml_encoding["plates"]
        kwargs["parents"] = [
            bayesnet.get_node_object(p) for p in yaml_encoding["parents"]
        ]
        kwargs["device"] = bayesnet.device
        if "observed" in yaml_encoding:
            kwargs["observed"] = yaml_encoding["observed"]
        return kwargs

    @classmethod
    def from_yaml_encoding(
        cls, name, bayesnet, yaml_encoding: Mapping[str, Any]
    ) -> "Node":
        kwargs = cls.process_yaml_spec(bayesnet, yaml_encoding)
        kwargs["name"] = name
        return cls(**kwargs)


class NodeWithCategoricalParents(Node):
    def __init__(
        self,
        name: str,
        plates: List[str],
        parents: Sequence[Node],
        **kwargs,
    ):
        if parents is None:
            parents = []
        self.parent_domain_sizes = torch.Size([p.domain_size for p in parents])
        super().__init__(name, plates, parents, **kwargs)

        assert all([p.value_type() == NodeValueType.CATEGORICAL for p in parents])

        # context specific independence
        self.model_csi = self._init_csi(self.parent_domain_sizes)
        self.guide_csi = self._init_csi(self.parent_domain_sizes)

    def _init_csi(
        self, size: torch.Size
    ) -> Optional[TensorIndexMapping]:  # pragma: no cover

        """Create initial structure for storing context specific independence."""
        if len(size) > 0:
            return TensorIndexMapping(size, self.device)
        else:
            return None

    def _add_csi_rule(self, csi, partial_mapping):
        assert csi is not None
        assert len(partial_mapping) <= len(self.parent_domain_sizes)
        indices = tuple(
            partial_mapping.get(d, slice(None, None, None))
            for d in range(len(self.parent_domain_sizes))
        )
        canonical_index = tuple(
            partial_mapping.get(d, 0) for d in range(len(self.parent_domain_sizes))
        )
        csi.add_mapping(indices, canonical_index)

    def model_add_csi_rule(self, partial_mapping: Mapping[int, int]) -> None:
        """Add context specific independence rule to model CPD.

        :param partial_mapping: context for the csi rule
        """

        self._add_csi_rule(self.model_csi, partial_mapping)

    def guide_add_csi_rule(self, partial_mapping: Mapping[int, int]) -> None:
        """Add context specific independence rule to guide CPD.

        :param partial_mapping: context for the csi rule
        """

        self._add_csi_rule(self.guide_csi, partial_mapping)

    def _index_cpd(
        self, cpd: torch.Tensor, *parent_values: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Index the conditional probability distribution of the node using parent values and return it.

        :param cpd: tensor representing the CPD of the node
        :param parent_values: tensors containing values of the parents of the node
        """
        assert len(parent_values) == len(self.parent_domain_sizes)
        if len(parent_values) > 0:
            t_parent_values = tuple(p.long() for p in parent_values)
            return Vindex(cpd)[t_parent_values]
        else:
            return cpd

    def get_sample_dist(
        self, sampling_mode: SamplingMode, *parent_values: torch.Tensor
    ) -> distributions.Distribution:
        """
        Get the CPD for this node based on its parents' values
        and return the corresponding distribution function.
        """

        # sample the appropriate cpd
        if sampling_mode == SamplingMode.MODEL:
            cpd = self._index_cpd(self.sample_model_cpd(), *parent_values)
        elif sampling_mode == SamplingMode.GUIDE:
            cpd = self._index_cpd(self.sample_guide_cpd(), *parent_values)
        elif sampling_mode == SamplingMode.MAP_POSTERIOR:
            cpd = self._index_cpd(self.MAP_cpd(), *parent_values)
        else:
            raise ValueError(f"unrecognized sampling mode- {sampling_mode}")

        return self.dist(cpd)
