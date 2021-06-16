from typing import Any, List, Mapping, Sequence, Tuple

import pyro
import pyro.distributions as distributions
import torch
import torch.distributions.constraints as constraints
from pyro.distributions.util import broadcast_shape
from pyro.ops.indexing import Vindex

from pearl.common import NodeValueType, SamplingMode
from pearl.nodes.bayesnetnode import Node, NodeWithCategoricalParents


class CategoricalNodeWithDirichletPrior(NodeWithCategoricalParents):
    """Categorically distributed node with Dirichlet prior."""

    def __init__(
        self,
        name: str,
        domain: Sequence[str],
        plates: List[str],
        parents: Sequence[Node],
        device: torch.device = torch.device("cpu"),
        observed: bool = True,
        prior_params: Mapping[str, torch.Tensor] = None,
    ) -> None:

        if len(domain) < 2:
            raise ValueError(
                f"Domain of categorical node {name} must include at-least two categories."
            )
        self.domain = domain

        super().__init__(
            name,
            plates,
            parents,
            observed=observed,
            device=device,
            prior_params=prior_params,
        )
        self.guide_MAP_cpd = None

    @property
    def domain_size(self):
        return len(self.domain)

    def value_type(self) -> NodeValueType:
        return NodeValueType.CATEGORICAL

    def dist(self, probs) -> distributions.Distribution:
        return distributions.Categorical(probs)

    def get_default_prior_params(self):
        return {
            "alpha": torch.ones(
                self.parent_domain_sizes + (self.domain_size,), device=self.device
            )
        }

    def validate_prior_params(self):
        super().validate_prior_params()
        assert self.prior_params["alpha"].shape == self.parent_domain_sizes + (
            self.domain_size,
        )
        assert self.prior_params["alpha"].gt(0).all()

    def sample_model_cpd(self) -> torch.Tensor:
        with pyro.plate_stack(f"{self.name}/cpd/plate", self.parent_domain_sizes):
            self.model_cpd = pyro.sample(
                f"{self.name}/cpd", distributions.Dirichlet(self.prior_params["alpha"])
            )
        if self.model_csi is not None:
            self.model_cpd = self.model_csi.map(self.model_cpd)
        return self.model_cpd

    def sample_guide_cpd(self) -> torch.Tensor:
        self.guide_alpha = pyro.param(
            f"{self.name}/guide/alpha",
            self.prior_params["alpha"],
            constraint=constraints.positive,
        )
        with pyro.plate_stack(f"{self.name}/cpd/plate", self.parent_domain_sizes):
            self.guide_cpd = pyro.sample(
                f"{self.name}/cpd", distributions.Dirichlet(self.guide_alpha)
            )
        if self.guide_csi is not None:
            self.guide_cpd = self.guide_csi.map(self.guide_cpd)
        return self.guide_cpd

    def MAP_cpd(self) -> torch.Tensor:
        """Return the MAP estimate of the model CPD."""
        self.guide_MAP_cpd = distributions.Dirichlet(self.guide_alpha).mean
        if self.guide_csi is not None:
            self.guide_MAP_cpd = self.guide_csi.map(self.guide_MAP_cpd)
        return self.guide_MAP_cpd

    def overwrite_param_store(self) -> None:
        param_store = pyro.get_param_store()
        assert self.guide_alpha is not None, "Guide hyperparameters don't exist"
        param_store["/".join((self.name, "guide", "alpha"))] = self.guide_alpha

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        self.sample_guide_cpd()  # to ensure that guide parameters are set
        yaml_encoding = dict(super().to_yaml_encoding())
        yaml_encoding["domain"] = list(self.domain)
        yaml_encoding["prior_params"] = {"alpha": self.guide_alpha.tolist()}
        return yaml_encoding

    @classmethod
    def process_yaml_spec(
        cls, bayesnet, yaml_encoding: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        kwargs = super().process_yaml_spec(bayesnet, yaml_encoding)
        kwargs["domain"] = yaml_encoding["domain"]
        if "prior_params" in yaml_encoding:
            kwargs["prior_params"] = {
                "alpha": torch.tensor(
                    yaml_encoding["prior_params"]["alpha"], device=bayesnet.device
                )
            }
        return kwargs


class GeneralizedLinearNode(Node):
    """
    Node which implements the generlized linear model for sampling a
    categorical random variable conditioned on a mix of categorical
    and continuous parents.

    This node requires atleast one continuous parent.  For each
    combination of discrete parents, the random variable is sampled
    from a categorical distribution that is computed using a softmax
    function applied to a linear combination of the continuous
    parents.

    If the discrete parents have domains of size D1,...,Dk, the
    number of categories are M, and the number of continuous parents
    are C, then it uses 'bias' and 'weight' tensors of shape [D1, ...,
    Dk, M] and [D1, ..., Dk, M, C].  Instead of using point estimates,
    these tensors are sampled from normal priors over them.
    """

    def __init__(
        self,
        name: str,
        domain: Sequence[str],
        plates: List[str],
        parents: Sequence[Node],
        device: torch.device = torch.device("cpu"),
        prior_params: Mapping[str, torch.Tensor] = None,
        observed: bool = True,
    ) -> None:
        self.domain = domain

        # record positions of discrete and continuous parents
        discrete_parent_indices = []
        continuous_parent_indices = []
        for i, p in enumerate(parents):
            if p.value_type() == NodeValueType.CATEGORICAL:
                discrete_parent_indices.append(i)
            elif p.value_type() == NodeValueType.CONTINUOUS:
                continuous_parent_indices.append(i)
            else:
                raise ValueError(
                    f"unsupported parent of type {p.value_type()} for node {name}"
                )
        assert (
            len(continuous_parent_indices) > 0
        ), f"GeneralizedLinearNode {name} requires atleast one continuous parent"
        self.discrete_parent_indices = discrete_parent_indices
        self.continuous_parent_indices = continuous_parent_indices

        # get domain sizes of discrete parents
        discrete_parent_domain_sizes = []
        for i in self.discrete_parent_indices:
            discrete_parent_domain_sizes.append(parents[i].domain_size)
        self.discrete_parent_domain_sizes = torch.Size(discrete_parent_domain_sizes)

        super().__init__(
            name=name,
            plates=plates,
            parents=parents,
            device=device,
            prior_params=prior_params,
            observed=observed,
        )

    @property
    def domain_size(self):
        return len(self.domain)

    def value_type(self) -> NodeValueType:
        return NodeValueType.CATEGORICAL

    def dist(self, probs) -> distributions.Distribution:
        return distributions.Categorical(probs)

    def get_default_prior_params(self):
        return {
            "bias_mean": torch.zeros(
                *self.discrete_parent_domain_sizes, len(self.domain)
            ),
            "bias_scale": torch.ones(
                *self.discrete_parent_domain_sizes, len(self.domain)
            ),
            "weights_mean": torch.zeros(
                *self.discrete_parent_domain_sizes,
                len(self.continuous_parent_indices),
                len(self.domain),
            ),
            "weights_scale": torch.ones(
                *self.discrete_parent_domain_sizes,
                len(self.continuous_parent_indices),
                len(self.domain),
            ),
        }

    def validate_prior_params(self):
        super().validate_prior_params()
        assert self.prior_params["bias_mean"].shape == (
            *self.discrete_parent_domain_sizes,
            len(self.domain),
        )
        assert self.prior_params["bias_scale"].shape == (
            *self.discrete_parent_domain_sizes,
            len(self.domain),
        )
        assert self.prior_params["weights_mean"].shape == (
            *self.discrete_parent_domain_sizes,
            len(self.continuous_parent_indices),
            len(self.domain),
        )
        assert self.prior_params["weights_scale"].shape == (
            *self.discrete_parent_domain_sizes,
            len(self.continuous_parent_indices),
            len(self.domain),
        )

    def sample_model_cpd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with pyro.plate_stack(
            f"{self.name}/cpd/bias/plate", self.prior_params["bias_mean"].shape
        ):
            bias = pyro.sample(
                f"{self.name}/bias",
                distributions.Normal(
                    self.prior_params["bias_mean"],
                    self.prior_params["bias_scale"],
                ),
            )

        with pyro.plate_stack(
            f"{self.name}/cpd/weights/plate",
            self.prior_params["weights_mean"].shape,
        ):
            weights = pyro.sample(
                f"{self.name}/weights",
                distributions.Normal(
                    self.prior_params["weights_mean"],
                    self.prior_params["weights_scale"],
                ),
            )
        self.model_cpd = (weights, bias)
        return self.model_cpd

    def sample_guide_cpd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.guide_bias_mean = pyro.param(
            f"{self.name}/guide/bias_mean",
            self.prior_params["bias_mean"],
        )
        self.guide_bias_scale = pyro.param(
            f"{self.name}/guide/bias_scale",
            self.prior_params["bias_scale"],
            constraint=constraints.positive,
        )
        self.guide_weights_mean = pyro.param(
            f"{self.name}/guide/weights_mean",
            self.prior_params["weights_mean"],
        )
        self.guide_weights_scale = pyro.param(
            f"{self.name}/guide/weights_scale",
            self.prior_params["weights_scale"],
            constraint=constraints.positive,
        )
        with pyro.plate_stack(
            f"{self.name}/cpd/bias/plate", self.guide_bias_mean.shape
        ):
            bias = pyro.sample(
                f"{self.name}/bias",
                distributions.Normal(
                    self.guide_bias_mean,
                    self.guide_bias_scale,
                ),
            )
        with pyro.plate_stack(
            f"{self.name}/cpd/weights/plate", self.guide_weights_mean.shape
        ):
            weights = pyro.sample(
                f"{self.name}/weights",
                distributions.Normal(
                    self.guide_weights_mean,
                    self.guide_weights_scale,
                ),
            )
        self.guide_cpd = (weights, bias)
        return self.guide_cpd

    def MAP_cpd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.guide_weights_mean, self.guide_bias_mean)

    def _index_cpd(
        self, cpd: torch.Tensor, *parent_values: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Index the conditional probability distribution of the node
        using parent values and return it.

        :param cpd: tensor representing the CPD of the node
        :param parent_values: tensors containing values of the parents
            of the node
        """
        assert len(parent_values) == len(self.discrete_parent_indices)
        if len(parent_values) > 0:
            t_parent_values = tuple(p.long() for p in parent_values)
            return Vindex(cpd)[t_parent_values]
        else:
            return cpd

    def get_sample_dist(
        self, sampling_mode: SamplingMode, *parent_values: torch.Tensor
    ) -> distributions.Distribution:
        if sampling_mode == SamplingMode.MODEL:
            weights, bias = self.sample_model_cpd()
        elif sampling_mode == SamplingMode.GUIDE:
            weights, bias = self.sample_guide_cpd()
        elif sampling_mode == SamplingMode.MAP_POSTERIOR:
            weights, bias = self.MAP_cpd()
        else:
            raise ValueError(f"uncrecognized sampling mode- {sampling_mode}")

        # index the cpd based on values of discrete parents
        discrete_parent_values = [
            parent_values[i] for i in self.discrete_parent_indices
        ]
        weights = self._index_cpd(weights, *discrete_parent_values)
        bias = self._index_cpd(bias, *discrete_parent_values)

        # stack continuous parent values after broadcasting them
        continuous_parent_shapes = [
            parent_values[i].shape for i in self.continuous_parent_indices
        ]
        broadcasted_shape = broadcast_shape(*continuous_parent_shapes)
        stacked_continuous_parent_tensors = torch.stack(
            [
                parent_values[i].expand(broadcasted_shape)
                for i in self.continuous_parent_indices
            ],
            dim=-1,
        )
        stacked_continuous_parent_tensors.unsqueeze_(
            -1
        )  # add a dimension for categories
        probs = torch.softmax(
            torch.sum(stacked_continuous_parent_tensors * weights, dim=-2) + bias,
            dim=-1,
        )
        return self.dist(probs)

    def overwrite_param_store(self) -> None:
        param_store = pyro.get_param_store()
        param_store[
            "/".join((self.name, "guide", "weights_mean"))
        ] = self.guide_weights_mean
        param_store[
            "/".join((self.name, "guide", "weights_scale"))
        ] = self.guide_weights_scale
        param_store["/".join((self.name, "guide", "bias_mean"))] = self.guide_bias_mean
        param_store[
            "/".join((self.name, "guide", "bias_scale"))
        ] = self.guide_bias_scale

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        self.sample_guide_cpd()
        yaml_encoding = dict(super().to_yaml_encoding())
        yaml_encoding["domain"] = list(self.domain)
        yaml_encoding["prior_params"] = {
            "weights_mean": self.guide_weights_mean.tolist(),
            "weights_scale": self.guide_weights_scale.tolist(),
            "bias_mean": self.guide_bias_mean.tolist(),
            "bias_scale": self.guide_bias_scale.tolist(),
        }
        return yaml_encoding

    @classmethod
    def process_yaml_spec(
        cls, bayesnet, yaml_encoding: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        kwargs = super().process_yaml_spec(bayesnet, yaml_encoding)
        kwargs["domain"] = yaml_encoding["domain"]
        if "prior_params" in yaml_encoding:
            kwargs["prior_params"] = {
                "weights_mean": torch.tensor(
                    yaml_encoding["prior_params"]["weights_mean"],
                    device=bayesnet.device,
                ),
                "weights_scale": torch.tensor(
                    yaml_encoding["prior_params"]["weights_scale"],
                    device=bayesnet.device,
                ),
                "bias_mean": torch.tensor(
                    yaml_encoding["prior_params"]["bias_mean"], device=bayesnet.device
                ),
                "bias_scale": torch.tensor(
                    yaml_encoding["prior_params"]["bias_scale"], device=bayesnet.device
                ),
            }
        return kwargs
