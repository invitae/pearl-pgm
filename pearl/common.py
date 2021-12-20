import logging
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple

import pyro
import torch


@dataclass
class Plate:
    """
    Class to represent plates in graphical models.
    Stores corresponding pyro plate object.
    """

    name: str
    dim: int

    # underlying pyro plate
    pyro_plate: pyro.plate = None


class SamplingMode(Enum):
    MODEL = 0
    GUIDE = 1
    MAP_POSTERIOR = 2


class NodeValueType(IntEnum):
    CATEGORICAL = 0
    CONTINUOUS = 1


class QueryType(Enum):
    CONJUNCTIVE = 0
    DISJUNCTIVE = 1


def same_device(device1, device2) -> bool:  # pragma: no cover
    if device1.type == device2.type == "cpu":
        return True
    elif device1.type == device2.type == "cuda" and (device1.index == device2.index):
        return True
    else:
        return False


def _all_index_combinations(domain_sizes: torch.Size) -> torch.Tensor:

    """Enumerate the possible combinations of indices for "domain_sizes". e.g. if
    the node has 2 parents with 2 and 3 possible values respectively, returns
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

    reshape() is required to ensure the return value has the correct dims in
    the case where there is a single domain size.
    """
    return torch.cartesian_prod(*(torch.arange(s) for s in domain_sizes)).reshape(
        torch.prod(torch.tensor(domain_sizes)), len(domain_sizes)
    )


def _all_index_combinations_tuple(domain_sizes: torch.Size) -> Tuple[torch.Tensor]:
    index_combinations = _all_index_combinations(domain_sizes)
    return tuple(
        t.squeeze()
        for t in torch.chunk(index_combinations, chunks=len(domain_sizes), dim=-1)
    )


def tensor_accuracy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    if not tensor1.shape == tensor2.shape:
        raise ValueError(
            f"Shapes of input tensors are not equal: {tensor1.shape} vs {tensor2.shape}"
        )

    return float(
        torch.sum(tensor1 == tensor2) / torch.prod(torch.tensor(tensor1.shape)).float()
    )


def get_logger(name, level=None):
    logger = logging.getLogger(name)
    level = level or logging.INFO
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger
