from typing import Any, Callable, List, Mapping, Sequence

import pyro.distributions as dist
import torch

from pearl.common import NodeValueType, SamplingMode
from pearl.nodes.bayesnetnode import Node


class DeterministicNode(Node):
    """Node whose value is a deterministic function of its parents,
    as defined by the func input parameter.

    func must take in a Dict[str, torch.tensor] with
    keys corresponding to parent names and values to parent values
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        node_value_type: NodeValueType,
        plates: List[str],
        parents: Sequence[Node],
        domain: Sequence[str] = None,
        device: torch.device = torch.device("cpu"),
        observed: bool = False,
    ) -> None:
        super().__init__(
            name,
            plates,
            parents,
            observed=observed,
            device=device,
        )

        if node_value_type == NodeValueType.CATEGORICAL and domain is None:
            raise ValueError(
                f"domain must be set for categorical deterministic node {self.name}"
            )
        if node_value_type == NodeValueType.CONTINUOUS and domain is not None:
            raise ValueError(
                f"domain should not be set for continuous deterministic node {self.name}"
            )

        self.observed = observed
        self.node_value_type = node_value_type
        self.domain = domain
        self.func = func

    @property
    def domain_size(self):
        return len(self.domain) if self.domain else None

    def get_sample_dist(
        self, sampling_mode: SamplingMode, *parent_values: torch.Tensor
    ) -> dist.Distribution:
        """
        Compute the deterministic function based on the parents' values
        and return the corresponding Delta distribution.
        """

        d_parent_values = {
            parent.name: value for parent, value in zip(self.parents, parent_values)
        }

        f_output = self.func(d_parent_values)

        return dist.Delta(f_output)

    def value_type(self) -> NodeValueType:
        return self.node_value_type

    def overwrite_param_store(self) -> None:
        pass

    @classmethod
    def process_yaml_spec(
        cls, bayesnet, yaml_encoding: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        kwargs = super().process_yaml_spec(bayesnet, yaml_encoding)
        if "domain" in yaml_encoding:
            kwargs["domain"] = yaml_encoding["domain"]
        return kwargs


class Exponential(DeterministicNode):
    """
    Node which implements the 'torch.exp' function
    """

    def __init__(
        self,
        name: str,
        plates: List[str],
        parents: Sequence[Node],
        device: torch.device = torch.device("cpu"),
        observed: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            func=torch.exp,
            node_value_type=NodeValueType.CONTINUOUS,
            plates=plates,
            parents=parents,
            device=device,
            observed=observed,
        )

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        yaml_encoding = dict(super().to_yaml_encoding())
        return yaml_encoding


class Sum(DeterministicNode):
    """
    Node which implements the 'torch.sum' function
    """

    def __init__(
        self,
        name: str,
        plates: List[str],
        parents: Sequence[Node],
        device: torch.device = torch.device("cpu"),
        observed: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            func=torch.sum,
            node_value_type=NodeValueType.CONTINUOUS,
            plates=plates,
            parents=parents,
            device=device,
            observed=observed,
        )

    def to_yaml_encoding(self) -> Mapping[str, Any]:
        yaml_encoding = dict(super().to_yaml_encoding())
        return yaml_encoding
