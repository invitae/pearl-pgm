from typing import Any, List, Mapping, Sequence, Tuple

import pyro
import pyro.distributions as distributions
import torch
import torch.distributions.constraints as constraints
from pyro.distributions.util import broadcast_shape
from pyro.ops.indexing import Vindex

from pearl.common import NodeValueType, SamplingMode
from pearl.nodes.bayesnetnode import Node, NodeWithCategoricalParents


class ContinuousNodeWithNormalDistribution(NodeWithCategoricalParents):
    """Node whose continuous value is drawn from a Normal distribution. The prior for the mean of the normal
    distribution is given by a normal distribution itself, specified with "mean_mean" and "mean_scale" prior
    parameters with defaults 0.0 and 1.0, while the prior for the scale of the normal distribution is given by a
    HalfCauchy distribution with scale "scale_scale", 1.0 by default. If any of the prior parameters are provided as
    individual floats then those values will be expanded to tensors matching parent domain sizes.
    """

    def __init__(
        self,
        name: str,
        plates: List[str],
        parents: Sequence[Node],
        device: torch.device = torch.device("cpu"),
        observed: bool = True,
        prior_params: Mapping[str, torch.Tensor] = None,
    ) -> None:

        super().__init__(
            name,
            plates,
            parents,
            observed=observed,
            device=device,
            prior_params=prior_params,
        )
        self.guide_MAP_cpd = None

    def get_default_prior_params(self):
        # The tuple value (1,) catted to the end of parent_domain_sizes ensures
        # that we have parameters for the distribution even if the node has no parents.
        return {
            # Mean of normal prior on mean.
            "mean_mean": torch.zeros(self.parent_domain_sizes, device=self.device),
            # Scale of normal prior on mean.
            "mean_scale": torch.ones(self.parent_domain_sizes, device=self.device),
            # Scale of half-cauchy prior on scale.
            "scale_scale": torch.ones(self.parent_domain_sizes, device=self.device),
        }

    def value_type(self) -> NodeValueType:
        return NodeValueType.CONTINUOUS

    def dist(self, loc_and_scale) -> distributions.Distribution:
        return distributions.Normal(
            loc_and_scale[Ellipsis, 0], loc_and_scale[Ellipsis, 1]
        )

    def validate_prior_params(self):
        super().validate_prior_params()
        for param_name in ["mean_mean", "mean_scale", "scale_scale"]:
            assert self.prior_params[param_name].shape == self.parent_domain_sizes

        for param_name in ["mean_scale", "scale_scale"]:
            assert self.prior_params[param_name].gt(0).all()

    def sample_model_cpd(self) -> None:
        with pyro.plate_stack(f"{self.name}/cpd/plate", self.parent_domain_sizes):
            mean = pyro.sample(
                f"{self.name}/cpd/mean",
                distributions.Normal(
                    self.prior_params["mean_mean"], self.prior_params["mean_scale"]
                ),
            )

            scale = pyro.sample(
                f"{self.name}/cpd/scale",
                distributions.HalfCauchy(self.prior_params["scale_scale"]),
            )

            if self.model_csi is not None:
                mean = self.model_csi.map(mean)
                scale = self.model_csi.map(scale)

            self.model_cpd = torch.stack((mean, scale), dim=-1)
        return self.model_cpd

    def sample_guide_cpd(self) -> None:
        self.guide_mean_mean = pyro.param(
            f"{self.name}/guide/mean_mean", self.prior_params["mean_mean"]
        )
        self.guide_mean_scale = pyro.param(
            f"{self.name}/guide/mean_scale",
            self.prior_params["mean_scale"],
            constraint=constraints.positive,
        )
        self.guide_scale = pyro.param(
            f"{self.name}/guide/scale",
            self.prior_params["scale_scale"],
            constraint=constraints.positive,
        )

        with pyro.plate_stack(f"{self.name}/cpd/plate", self.parent_domain_sizes):
            mean = pyro.sample(
                f"{self.name}/cpd/mean",
                distributions.Normal(self.guide_mean_mean, self.guide_mean_scale),
            )
            scale = pyro.sample(
                f"{self.name}/cpd/scale", distributions.Delta(self.guide_scale)
            )

            if self.guide_csi is not None:
                mean = self.guide_csi.map(mean)
                scale = self.guide_csi.map(scale)

            self.guide_cpd = torch.stack((mean, scale), dim=-1)
        return self.guide_cpd

    def MAP_cpd(self) -> torch.Tensor:
        guide_mean_mean = self.guide_mean_mean
        guide_scale = self.guide_scale
        if self.guide_csi is not None:
            guide_mean_mean = self.guide_csi.map(guide_mean_mean)
            guide_scale = self.guide_csi.map(guide_scale)
        self.guide_MAP_cpd = torch.stack(
            (guide_mean_mean, guide_scale), dim=-1
        ).squeeze()
        return self.guide_MAP_cpd

    def overwrite_param_store(self) -> None:
        param_store = pyro.get_param_store()

        params = (self.guide_mean_mean, self.guide_mean_scale, self.guide_scale)
        assert all(param is not None for param in params)

        param_store["/".join((self.name, "guide", "mean_mean"))] = self.guide_mean_mean
        param_store[
            "/".join((self.name, "guide", "mean_scale"))
        ] = self.guide_mean_scale
        param_store["/".join((self.name, "guide", "scale"))] = self.guide_scale

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        self.sample_guide_cpd()  # to ensure that guide parameters are set
        yaml_encoding = dict(super().to_yaml_encoding())
        yaml_encoding["prior_params"] = {
            "mean_mean": self.guide_mean_mean.tolist(),
            "mean_scale": self.guide_mean_scale.tolist(),
            "scale_scale": self.guide_scale.tolist(),
        }
        return yaml_encoding

    @classmethod
    def process_yaml_spec(
        cls, bayesnet, yaml_encoding: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        kwargs = super().process_yaml_spec(bayesnet, yaml_encoding)
        if "prior_params" in yaml_encoding:
            kwargs["prior_params"] = {
                "mean_mean": torch.tensor(
                    yaml_encoding["prior_params"]["mean_mean"],
                    device=bayesnet.device,
                ),
                "mean_scale": torch.tensor(
                    yaml_encoding["prior_params"]["mean_scale"],
                    device=bayesnet.device,
                ),
                "scale_scale": torch.tensor(
                    yaml_encoding["prior_params"]["scale_scale"],
                    device=bayesnet.device,
                ),
            }
        return kwargs


class ConditionalLinearGaussianNode(Node):
    """
    Node which implements the conditional linear gaussian model for
    sampling a continuous random variable conditioned on a mix of
    categorical and continuous parents.

    This node requires atleast one continuous parent.  For each
    combination of discrete parents, the random variable is sampled
    from a Normal distribution whose mean is the linear combination of
    its continuous parents and a variance that is independent of
    parent values.

    If the discrete parents have domains of size D1, ..., Dk, and
    there are C continuous parents, then it uses 'bias' and 'weight'
    tensors of shape [D1, ..., Dk] and [D1, ..., Dk, C].  Instead of
    using point estimates, these tensors are sampled from normal
    priors over them.  Further the variant is also sampled from a
    half-cauchy prior.
    """

    def __init__(
        self,
        name: str,
        plates: List[str],
        parents: Sequence[Node],
        device: torch.device = torch.device("cpu"),
        prior_params: Mapping[str, torch.Tensor] = None,
        observed: bool = True,
    ) -> None:
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
        ), f"ConditionalLinearGaussianNode {name} requires atleast one continuous parent"
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

    def value_type(self) -> NodeValueType:
        return NodeValueType.CONTINUOUS

    def dist(self, loc, scale) -> distributions.Distribution:
        return distributions.Normal(loc, scale)

    def get_default_prior_params(self):
        return {
            "bias_mean": torch.full(self.discrete_parent_domain_sizes, 0.0),
            "bias_scale": torch.full(self.discrete_parent_domain_sizes, 1.0),
            "weights_mean": torch.zeros(
                *self.discrete_parent_domain_sizes, len(self.continuous_parent_indices)
            ),
            "weights_scale": torch.ones(
                *self.discrete_parent_domain_sizes, len(self.continuous_parent_indices)
            ),
            "scale_scale": torch.full(self.discrete_parent_domain_sizes, 1.0),
        }

    def validate_prior_params(self):
        super().validate_prior_params()
        assert self.prior_params["bias_mean"].shape == (
            *self.discrete_parent_domain_sizes,
        )
        assert self.prior_params["bias_scale"].shape == (
            *self.discrete_parent_domain_sizes,
        )
        assert self.prior_params["weights_mean"].shape == (
            *self.discrete_parent_domain_sizes,
            len(self.continuous_parent_indices),
        )
        assert self.prior_params["weights_scale"].shape == (
            *self.discrete_parent_domain_sizes,
            len(self.continuous_parent_indices),
        )
        assert self.prior_params["scale_scale"].shape == (
            *self.discrete_parent_domain_sizes,
        )

    def sample_model_cpd(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            f"{self.name}/cpd/weights/plate", self.prior_params["weights_mean"].shape
        ):
            weights = pyro.sample(
                f"{self.name}/weights",
                distributions.Normal(
                    self.prior_params["weights_mean"],
                    self.prior_params["weights_scale"],
                ),
            )
        with pyro.plate_stack(
            f"{self.name}/cpd/scale/plate", self.prior_params["scale_scale"].shape
        ):
            scale = pyro.sample(
                f"{self.name}/scale",
                distributions.HalfCauchy(self.prior_params["scale_scale"]),
            )
        self.model_cpd = (weights, bias, scale)
        return self.model_cpd

    def sample_guide_cpd(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.guide_scale_scale = pyro.param(
            f"{self.name}/guide/scale_scale",
            self.prior_params["scale_scale"],
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
        with pyro.plate_stack(
            f"{self.name}/cpd/scale/plate", self.guide_scale_scale.shape
        ):
            scale = pyro.sample(
                f"{self.name}/scale",
                distributions.Delta(self.guide_scale_scale),
            )
        self.guide_cpd = (weights, bias, scale)
        return self.guide_cpd

    def MAP_cpd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.guide_weights_mean, self.guide_bias_mean, self.guide_scale_scale)

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
            weights, bias, scale = self.sample_model_cpd()
        elif sampling_mode == SamplingMode.GUIDE:
            weights, bias, scale = self.sample_guide_cpd()
        elif sampling_mode == SamplingMode.MAP_POSTERIOR:
            weights, bias, scale = self.MAP_cpd()
        else:
            raise ValueError(f"unrecognized sampling mode- {sampling_mode}")

        # index the cpd based on values of discrete parents
        discrete_parent_values = [
            parent_values[i] for i in self.discrete_parent_indices
        ]
        weights = self._index_cpd(weights, *discrete_parent_values)
        bias = self._index_cpd(bias, *discrete_parent_values)
        scale = self._index_cpd(scale, *discrete_parent_values)

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
        mean = torch.sum(stacked_continuous_parent_tensors * weights, dim=-1) + bias
        return self.dist(mean, scale)

    def overwrite_param_store(self) -> None:
        param_store = pyro.get_param_store()
        param_store[
            "/".join((self.name, "guide", "weights_mean")),
        ] = self.guide_weights_mean
        param_store[
            "/".join((self.name, "guide", "weights_scale")),
        ] = self.guide_weights_scale
        param_store["/".join((self.name, "guide", "bias_mean"))] = self.guide_bias_mean
        param_store[
            "/".join((self.name, "guide", "bias_scale"))
        ] = self.guide_bias_scale
        param_store[
            "/".join((self.name, "guide", "scale_scale"))
        ] = self.guide_scale_scale

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        self.sample_guide_cpd()
        yaml_encoding = dict(super().to_yaml_encoding())
        yaml_encoding["prior_params"] = {
            "weights_mean": self.guide_weights_mean.tolist(),
            "weights_scale": self.guide_weights_scale.tolist(),
            "bias_mean": self.guide_bias_mean.tolist(),
            "bias_scale": self.guide_bias_scale.tolist(),
            "scale_scale": self.guide_scale_scale.tolist(),
        }
        return yaml_encoding

    @classmethod
    def process_yaml_spec(
        cls, bayesnet, yaml_encoding: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        kwargs = super().process_yaml_spec(bayesnet, yaml_encoding)
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
                    yaml_encoding["prior_params"]["bias_mean"],
                    device=bayesnet.device,
                ),
                "bias_scale": torch.tensor(
                    yaml_encoding["prior_params"]["bias_scale"],
                    device=bayesnet.device,
                ),
                "scale_scale": torch.tensor(
                    yaml_encoding["prior_params"]["scale_scale"],
                    device=bayesnet.device,
                ),
            }
        return kwargs
