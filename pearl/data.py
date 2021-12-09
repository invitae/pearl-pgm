import copy
from collections.abc import Iterable
from functools import reduce
from itertools import accumulate
from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pearl.common import NodeValueType, same_device


class VariableData(NamedTuple):
    """

    VariableData stores the meta-data of variables as well as the values
    they take in a dataset.

        - value_type: an instance of NodeValueType used for specifying
        whether the variable is CATEGORICAL or CONTINUOUS

        - value: multi-dimensional tensor containing the values the
        variable takes in the dataset.  The tensor contains one
        dimension for each plate in which it occurs.  The left-most
        dimension of the tensor corresponds to the instance plate and
        the remaining dimensions correspond to the other plates in the
        model.

        - domain: A sequence of outcome names. This is used only for
        CATEGORICAL variables and should be None for CONTINUOUS
        variables.
    """

    value_type: NodeValueType
    value: torch.Tensor
    discrete_domain: Optional[Sequence[str]] = None


class BayesianNetworkDataset(Dataset):
    """
    A dataset suitable for use by the BayesianNetwork.  We assume that
    the variables have atleast one batch dimension (the one
    corresponding to the instances).  There can be more based on the
    plates in the graphical model.
    """

    def __init__(
        self,
        variable_dict: Mapping[str, VariableData],
    ) -> None:
        """
        :param variable_dict: a dictionary mapping variables to their metadata and values
        """

        num_instances, device = self._validate_variable_dict(variable_dict)
        self.variable_dict = dict(variable_dict)
        self.num_instances = num_instances
        self.device = device

    def _validate_variable_dict(self, variable_dict):
        for varname, variable in variable_dict.items():
            if not variable.value.ndim > 0:
                raise ValueError(f"{varname} should have atleast one plate dimension")
            if variable.value_type == NodeValueType.CATEGORICAL:
                if variable.discrete_domain is None:
                    raise ValueError(f"{varname} is categorical and should have domain")
            if variable.value_type == NodeValueType.CONTINUOUS:
                if variable.discrete_domain is not None:
                    raise ValueError(
                        f"{varname} is continuous and should not have domain"
                    )
            if variable.value.dtype != torch.float:
                raise ValueError(f"{varname} value tensor should be of type float")

        all_num_instances = [
            variable.value.size(dim=0) for variable in variable_dict.values()
        ]
        if len(set(all_num_instances)) > 1:
            raise ValueError(
                f"all variables should have equal number of instances - {all_num_instances}"
            )
        all_devices = [variable.value.device for variable in variable_dict.values()]
        if not all(same_device(d, all_devices[0]) for d in all_devices):
            raise ValueError(
                f"all variables should use same device for value tensors - {all_devices}"
            )
        return all_num_instances[0], all_devices[0]

    def __len__(self) -> int:
        """
        Return the number of instances in the dataset.
        """
        return self.num_instances

    def subseq(self, i: int, j: int = None) -> "BayesianNetworkDataset":
        """
        Return the instances in the range (i, j) as a BayesianNetworkDataset object.
        If 'j' is omitted we return the instances starting from i.
        """
        variable_dict = dict()
        for v in self.variable_dict:
            variable_dict[v] = VariableData(
                self.variable_dict[v].value_type,
                self.variable_dict[v].value[i:j].clone().detach(),
                self.variable_dict[v].discrete_domain,
            )
        return BayesianNetworkDataset(variable_dict)

    def __getitem__(self, key) -> torch.Tensor:
        """
        Get the value of a variable in the BayesianNetworkDataset.
        The key is of the form (var_name, *idx) where var_name is the
        name of the variable and idx is an optional sequence of
        indices into the value tensor.
        """
        if isinstance(key, str) and key in self.variable_dict:
            return self.variable_dict[key].value
        elif isinstance(key, tuple) and key[0] in self.variable_dict:
            return self.variable_dict[key[0]].value[key[1:]]
        else:
            raise KeyError(f"unrecognized key {key}")

    def __setitem__(self, key, values) -> None:
        """
        Set the value of a variable in the BayesianNetworkDataset.
        The key is of the form (var_name, *idx) where var_name is the
        name of the variable and idx is an optional sequence of
        indices into the value tensor.
        """
        if isinstance(key, str) and key in self.variable_dict:
            self.variable_dict[key].value[:] = values
        elif isinstance(key, tuple) and key[0] in self.variable_dict:
            self.variable_dict[key[0]].value[key[1:]] = values
        else:
            raise KeyError(f"unrecognized key {key}")

    def select(self, assignment) -> torch.Tensor:
        """
        Compute mask that selects rows which are consistent with the
        assignment.  This operation is only supported for datasets
        which have no plate dimension other than the one corresponding
        to instances.
        """
        assert all(variable.value.ndim == 1 for variable in self.variable_dict.values())
        mask = torch.ones(self.num_instances, device=self.device, dtype=torch.bool)
        for varname, value in assignment.items():
            assert varname in self.variable_dict
            if isinstance(value, str):
                assert (
                    self.variable_dict[varname].value_type == NodeValueType.CATEGORICAL
                )
                assert value in self.variable_dict[varname].discrete_domain
                mask = mask & torch.eq(
                    self.variable_dict[varname].value,
                    self.variable_dict[varname].discrete_domain.index(value),
                )
            elif isinstance(value, Iterable):
                # This is a hacky way to support selecting with torch methods,
                # e.g. {"continuous_var": (torch.gt, 0.0)}
                mask = mask & value[0](self.variable_dict[varname].value, *value[1:])
            else:  # pragma: no cover
                raise ValueError(f"Can't handle select value {value}")
        return mask

    def discrete_domain_size(self, varname) -> int:
        """
        Return the size of the domain of a discrete variable
        """
        return len(self.discrete_domain(varname))

    def discrete_domain(self, varname) -> Sequence[str]:
        """
        Return the domain of a discrete variable
        """
        assert varname in self.variable_dict
        assert self.variable_dict[varname].value_type == NodeValueType.CATEGORICAL
        return self.variable_dict[varname].discrete_domain

    def split(
        self, partitions: Union[Tuple[int, ...], Tuple[np.ndarray, ...]]
    ) -> Sequence["BayesianNetworkDataset"]:
        """
        Create a split of the dataset based on a tuple of sizes or
        indices.  If a tuple of sizes is given a random split is
        returned.  On the other hand if a tuple of index sets is
        given, the split uses exactly the specified index sets.
        """
        if isinstance(partitions[0], int):
            assert sum(partitions) == len(self)
            assert all([size > 0 for size in partitions])
            indices = torch.randperm(sum(partitions)).tolist()
            index_sets = []
            for offset, length in zip(accumulate(partitions), partitions):
                index_sets.append(indices[offset - length : offset])
        elif isinstance(partitions[0], np.ndarray):
            assert all([p.ndim == 1 for p in partitions])
            assert reduce(np.intersect1d, partitions).size == 0
            assert np.array_equal(
                reduce(np.union1d, partitions),
                np.arange(len(self)),
            )
            index_sets = partitions
        else:
            raise ValueError(
                "partitions should be tuple of ints of tuple of numpy arrays"
            )

        subsets = []
        for index_set in index_sets:
            subset_variables_dict = dict()
            for varname, variable in self.variable_dict.items():
                subset_variables_dict[varname] = VariableData(
                    variable.value_type,
                    variable.value[index_set],
                    variable.discrete_domain,
                )
            subsets.append(BayesianNetworkDataset(subset_variables_dict))
        return subsets

    def project(self, variables: List[str]) -> "BayesianNetworkDataset":
        """
        Project the dataset onto the specified variables.  The
        resulting dataset contains the union of the plates of these
        variables.
        """
        assert len(variables) > 0
        assert all([v in self.variable_dict for v in variables])
        projected_variable_dict = {v: self.variable_dict[v] for v in variables}
        return BayesianNetworkDataset(
            variable_dict=projected_variable_dict,
        )

    def copy(self) -> "BayesianNetworkDataset":
        """Return a shallow copy"""
        return copy.copy(self)

    def deepcopy(self) -> "BayesianNetworkDataset":
        """Return a deep copy"""
        return copy.deepcopy(self)

    def is_categorical(self, var: str) -> bool:
        """
        Return True if the specified variable is a categorical
        variable in the dataset.
        """
        return self.variable_dict[var].value_type == NodeValueType.CATEGORICAL

    def to_hdf5(self, fname: str) -> None:
        """
        Save to hdf5 file.  This method converts all tensors to numpy
        arrays and stores them in the hdf5 file together with
        metadata.  Device information and id tensors are not retained
        when saving to hdf5.
        """
        with h5py.File(fname, "w") as hdf:
            for k, v in self.variable_dict.items():
                hdf5_dataset = hdf.create_dataset(k, data=v.value.cpu().numpy())
                hdf5_dataset.attrs["type"] = v.value_type
                if v.value_type == NodeValueType.CATEGORICAL:
                    hdf5_dataset.attrs["domain"] = v.discrete_domain

    @classmethod
    def from_hdf5(cls, fname: str) -> "BayesianNetworkDataset":
        """
        Instantiate a BayesianNetworkDataset object from an hdf5 file.
        Since device information and id tensors aren't stored when a
        BayesianNetworkDataset object is saved to hdf5 file, these get
        default values.  The user can use .cuda() method if the
        tensors need to moved to a cuda device.
        """
        variable_dict = dict()
        with h5py.File(fname, "r") as hdf:
            for k in hdf.keys():
                hdf5_dataset = hdf.get(k)
                if hdf5_dataset.attrs["type"] == NodeValueType.CATEGORICAL:
                    variable_dict[k] = VariableData(
                        NodeValueType.CATEGORICAL,
                        torch.tensor(hdf5_dataset).float(),
                        hdf5_dataset.attrs["domain"].tolist(),
                    )
                else:
                    variable_dict[k] = VariableData(
                        NodeValueType.CONTINUOUS,
                        torch.tensor(hdf5_dataset).float(),
                        None,
                    )
        return cls(variable_dict)

    def __eq__(self, other: "BayesianNetworkDataset") -> bool:
        """
        Test equality of two BayesianNetworkDataset objects.
        """
        if not same_device(self.device, other.device):
            return False
        if set(self.variable_dict.keys()) != set(other.variable_dict.keys()):
            return False
        for k, v in self.variable_dict.items():
            if v.value_type != other.variable_dict[k].value_type:
                return False
            if v.discrete_domain != other.variable_dict[k].discrete_domain:
                return False
            if not torch.equal(self[k], other[k]):
                return False
        return True

    def to(self, device: torch.device) -> "BayesianNetworkDataset":
        """
        Return a copy of the ``BayesianNetworkDataset`` object with
        tensors copied to the specified device.  If the object is
        already using the specified device a shallow copy is returned.

        :param device: torch device to be used for the tensors of the
            copied ``BayesianNetworkDataset`` object.
        """
        if same_device(self.device, device):
            return self.copy()
        else:
            variable_dict = dict()
            for k, v in self.variable_dict.items():
                value_copy = v.value.to(device)
                variable_dict[k] = VariableData(
                    v.value_type,
                    value_copy,
                    v.discrete_domain,
                )
            return BayesianNetworkDataset(
                variable_dict,
            )

    def cpu(self) -> "BayesianNetworkDataset":
        """
        Return a copy of the ``BayesianNetworkDataset`` object with
        tensors copied to CPU memory.  If the object is already using
        CPU memory a shallow copy is returned.
        """
        return self.to(torch.device("cpu", 0))

    def cuda(self, device: torch.device) -> "BayesianNetworkDataset":
        """
        Return a copy of the ``BayesianNetworkDataset`` object with
        tensors copied to the specified CUDA device.  If the object is
        already using the specified device a shallow copy is returned.

        :param device: torch cuda device to be used to store the
            tensors of the copied ``BayesianNetworkDataset`` object.
        """

        if not device.type == "cuda":
            raise ValueError(f"did not receive cuda device as input: {device}")
        return self.to(device)
