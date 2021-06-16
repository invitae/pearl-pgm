from typing import Tuple, Union

import torch

from pearl.common import _all_index_combinations_tuple


class TensorIndexMapping:
    """A mapping of indices to other indices (of a tensor).

    A many-to-one mapping of tensor indices serves the purpose of
    capturing context specific independence.  For example if A, B are
    parents of C.  Then there are four different distributions of C
    conditioned on the values of A, B (if all are binary variables).
    But if the value of B irrelevant when A is 1, then {A=1, B=0} and
    {A=1, B=1} can both be mapped to {A=1, B=0}.  In terms of tensor
    indices the indices [1,0] and [1,1] are both mapped to [1,0].
    """

    # TODO: rename size to something more intuitive
    def __init__(self, size: torch.Size, device: torch.device = None) -> None:
        """Initially we map each index to itself.

        :param size: size of the tensor for which we are maintaining
            the mapping
        :param device: device on which the mapping should be stored
        """

        assert len(size) > 0
        assert all([s > 1 for s in size])
        if device is None:
            device = torch.device("cpu")
        index_ranges = map(lambda s: torch.arange(s, device=device), size)
        self.mapping = torch.cartesian_prod(*index_ranges).reshape(*size, len(size))
        self.size = size

    def add_mapping(
        self, input_indices: Tuple[Union[int, slice]], output_index: Tuple[int]
    ) -> None:
        """Map input indices to the output index.

        :param input_indices: the range of indices to be mapped
        :param output_index: a single canonical index
        """

        assert len(input_indices) == len(output_index) == len(self.size)
        self.mapping[input_indices] = torch.tensor(
            output_index, device=self.mapping.device
        )

    def map(self, cpd: torch.Tensor) -> torch.Tensor:
        """
        Transform a conditional probability distribution tensor
        according to the mapping.

        :param cpd: tensor whose number/size of batch dimensions is
            suitable for indexing using self.mapping

        :return: transformed cpd
        """
        assert all(cpd.size(dim=i) == self.size[i] for i in range(len(self.size)))
        idx_tuple = _all_index_combinations_tuple(self.size)
        t_idx_tuple = tuple(
            t.view(self.size)
            for t in torch.chunk(self.mapping[idx_tuple], chunks=len(self.size), dim=-1)
        )
        t_cpd = cpd[t_idx_tuple]
        assert t_cpd.shape == cpd.shape
        return t_cpd
