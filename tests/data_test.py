from contextlib import contextmanager

import numpy as np
import pytest
import torch

from pearl.common import NodeValueType, same_device
from pearl.data import BayesianNetworkDataset, VariableData
from tests.markers import needs_cuda


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def mock_dataset():
    """
    Creates a mock dataset with four variables [a, b, c, d].  The
    variables are in plates [0], [0, 1], [0, 2], [0, 1, 2]
    respectively.  The plate 0 is the plate of instances in the
    dataset, while plates 1 and 2 are plates in the graphical model

    This fixture allows some variation in the form of changing number
    of instances and not having plates.
    """

    def _mock_dataset(length=10, with_plates=True, no_id=False):
        value_types = [
            NodeValueType.CATEGORICAL,
            NodeValueType.CATEGORICAL,
            NodeValueType.CATEGORICAL,
            NodeValueType.CONTINUOUS,
        ]
        if with_plates:
            values = [
                torch.zeros([length]),
                torch.zeros([length, 2]),
                torch.zeros([length, 3]),
                torch.zeros([length, 2, 3]),
            ]
        else:
            values = [torch.zeros([length]) for _ in range(4)]

        domains = [
            ["False", "True"],
            ["low", "medium", "high"],
            ["benign", "likely_benign", "vus", "likely_pathogenic", "pathogenic"],
            None,
        ]

        varnames = ["a", "b", "c", "d"]
        variable_descriptions = [
            VariableData(*data) for data in zip(value_types, values, domains)
        ]
        return BayesianNetworkDataset(
            variable_dict=dict(zip(varnames, variable_descriptions)),
        )

    return _mock_dataset


@pytest.mark.parametrize("no_id", [(False), (True)])
@pytest.mark.parametrize("length", [(1), (10), (100)])
def test_bayesian_network_dataset_len(length, no_id, mock_dataset):
    dataset = mock_dataset(length=length, no_id=no_id)
    assert len(dataset) == length


@pytest.mark.parametrize(
    "key, value, expected_exception",
    [
        ("foo", torch.rand(10), pytest.raises(KeyError)),
        ("a", torch.rand(10), does_not_raise()),
        (("a", slice(0, 5)), torch.rand(5), does_not_raise()),
        ("b", torch.rand(10, 2), does_not_raise()),
        (("b", slice(0, 5), slice(0, 1)), torch.rand(5, 1), does_not_raise()),
        ("c", torch.rand(10, 3), does_not_raise()),
        (("c", slice(0, 5), slice(0, 1)), torch.rand(5, 1), does_not_raise()),
        ("d", torch.rand(10, 2, 3), does_not_raise()),
        (
            ("d", slice(0, 5), slice(0, 1), slice(0, 2)),
            torch.rand(5, 1, 2),
            does_not_raise(),
        ),
        (
            ("d", slice(0, 5), slice(0, 1)),
            torch.rand(10, 1, 3),
            pytest.raises(RuntimeError),
        ),
    ],
)
def test_bayesian_network_dataset_set_get(key, value, expected_exception, mock_dataset):
    dataset = mock_dataset()
    with expected_exception:
        dataset[key] = value
        assert torch.equal(dataset[key], value)


def test_bayesian_network_dataset_select_with_plates(mock_dataset):
    dataset = mock_dataset()
    with pytest.raises(AssertionError):
        dataset.select({"a": "True"})


@pytest.mark.parametrize("num_rows", [0, 1, 10])
def test_bayesian_network_dataset_select_without_plates(num_rows, mock_dataset):
    dataset = mock_dataset(with_plates=False)
    for variable in dataset.variable_dict.values():
        variable.value[:num_rows] = 1
    assert dataset.select({"a": "True"}).sum() == num_rows
    assert dataset.select({"b": "medium"}).sum() == num_rows
    assert dataset.select({"c": "likely_benign"}).sum() == num_rows
    assert dataset.select({"d": (torch.gt, 0.0)}).sum() == num_rows


@pytest.mark.parametrize(
    "var, domain_size, expected_exception",
    [
        ("a", 2, does_not_raise()),
        ("b", 3, does_not_raise()),
        ("c", 5, does_not_raise()),
        ("d", None, pytest.raises(AssertionError)),
    ],
)
def test_bayesian_network_dataset_domain_size(
    var, domain_size, expected_exception, mock_dataset
):
    dataset = mock_dataset()
    with expected_exception:
        assert dataset.discrete_domain_size(var) == domain_size


@pytest.mark.parametrize(
    "length1, length2, expected_exception",
    [
        (1, 9, does_not_raise()),
        (5, 5, does_not_raise()),
        (9, 1, does_not_raise()),
        (0, 10, pytest.raises(AssertionError)),
        (5, 10, pytest.raises(AssertionError)),
    ],
)
def test_bayesian_network_dataset_split_using_sizes(
    length1, length2, expected_exception, mock_dataset
):
    dataset = mock_dataset()
    with expected_exception:
        part1, part2 = dataset.split((length1, length2))
        assert len(part1) == length1
        assert len(part2) == length2


@pytest.mark.parametrize(
    "idx1, idx2, expected_exception",
    [
        (np.array([0]), np.arange(1, 10), does_not_raise()),
        (np.arange(5), np.arange(5, 10), does_not_raise()),
        (np.arange(5), np.arange(5), pytest.raises(AssertionError)),
        (np.arange(5), np.arange(5, 9), pytest.raises(AssertionError)),
    ],
)
def test_bayesian_network_dataset_split_using_indices(
    idx1, idx2, expected_exception, mock_dataset
):
    dataset = mock_dataset()
    with expected_exception:
        part1, part2 = dataset.split((idx1, idx2))
        assert len(part1) == len(idx1)
        assert len(part2) == len(idx2)
        assert torch.equal(dataset["a"][idx1], part1["a"])
        assert torch.equal(dataset["a"][idx2], part2["a"])


@pytest.mark.parametrize(
    "columns, expected_exception",
    [
        ([], pytest.raises(AssertionError)),
        (["foo", "bar"], pytest.raises(AssertionError)),
        (["a"], does_not_raise()),
        (["a", "d"], does_not_raise()),
    ],
)
def test_bayesian_network_dataset_project(columns, expected_exception, mock_dataset):
    dataset = mock_dataset()
    with expected_exception:
        projected_dataset = dataset.project(columns)
        for c in columns:
            assert torch.equal(projected_dataset[c], dataset[c])


@pytest.mark.parametrize("i, j", [(0, 1), (0, None), (0, 10)])
def test_bayesian_network_dataset_subseq(mock_dataset, i, j):
    dataset = mock_dataset()
    for var in dataset.variable_dict:
        dataset[var] = torch.rand_like(dataset[var])

    subseq = dataset.subseq(i, j)
    for var in dataset.variable_dict:
        assert torch.equal(dataset[var][i:j], subseq[var])


def test_bayesian_network_dataset_creation_fails_with_no_plates(mocker):
    with pytest.raises(ValueError):
        _ = BayesianNetworkDataset(
            variable_dict={
                "foo": VariableData(NodeValueType.CONTINUOUS, torch.tensor(1.0))
            }
        )


def test_bayesian_network_dataset_creation_fails_with_mismatch_in_instances(mocker):
    with pytest.raises(ValueError):
        _ = BayesianNetworkDataset(
            variable_dict={
                "foo": VariableData(
                    NodeValueType.CATEGORICAL, torch.zeros([10]), ["False", "True"]
                ),
                "bar": VariableData(
                    NodeValueType.CATEGORICAL, torch.zeros([20]), ["False", "True"]
                ),
            }
        )


def test_bayesian_network_dataset_creation_fails_without_domain_for_categorical_variable(
    mocker,
):
    with pytest.raises(ValueError):
        _ = BayesianNetworkDataset(
            variable_dict={
                "foo": VariableData(NodeValueType.CATEGORICAL, torch.zeros([10]))
            }
        )


def test_bayesian_network_dataset_creation_fails_with_domain_for_continuous_variable(
    mocker,
):
    with pytest.raises(ValueError):
        _ = BayesianNetworkDataset(
            variable_dict={
                "foo": VariableData(
                    NodeValueType.CONTINUOUS, torch.zeros([10]), mocker.Mock()
                )
            }
        )


def test_bayesian_network_dataset_creation_fails_when_value_is_not_float_tensor(mocker):
    with pytest.raises(ValueError):
        _ = BayesianNetworkDataset(
            variable_dict={
                "foo": VariableData(NodeValueType.CONTINUOUS, torch.zeros(5).long())
            }
        )


def test_hdf5_roundtrip(tmp_path, mock_dataset):
    dataset = mock_dataset(no_id=True)
    hdf5_file = tmp_path / "dataset.h5"
    dataset.to_hdf5(hdf5_file)
    loaded_dataset = BayesianNetworkDataset.from_hdf5(hdf5_file)
    assert dataset == loaded_dataset


def test_hdf5_roundtrip_with_file_objects(tmp_path, mock_dataset):
    dataset = mock_dataset(no_id=True)
    hdf5_file = tmp_path / "dataset.h5"
    with open(hdf5_file, "wb") as f:
        dataset.to_hdf5(f)
    with open(hdf5_file, "rb") as f:
        loaded_dataset = BayesianNetworkDataset.from_hdf5(f)
    assert dataset == loaded_dataset


@needs_cuda
def test_device_change_support(mock_dataset):
    dataset = mock_dataset()

    copy_to_device = torch.device("cuda", 0)
    cuda_dataset = dataset.to(copy_to_device)
    assert same_device(cuda_dataset.device, copy_to_device)
    for k in cuda_dataset.variable_dict:
        assert same_device(cuda_dataset[k].device, copy_to_device)

    cpu_dataset = cuda_dataset.to(dataset.device)
    assert same_device(cpu_dataset.device, dataset.device)
    for k in cpu_dataset.variable_dict:
        assert same_device(cpu_dataset[k].device, dataset.device)


@needs_cuda
def test_cpu_cuda(mock_dataset):
    CPU = torch.device("cpu", 0)
    CUDA = torch.device("cuda", 0)

    dataset = mock_dataset()

    # cpu -> cpu copy
    cpu_dataset = dataset.cpu()
    assert same_device(cpu_dataset.device, CPU)

    # cpu -> cuda copy
    cuda_dataset = dataset.cuda(CUDA)
    assert same_device(cuda_dataset.device, CUDA)

    # cuda -> cuda copy
    another_cuda_dataset = cuda_dataset.cuda(CUDA)
    assert same_device(another_cuda_dataset.device, CUDA)

    # cuda -> cpu copy
    another_cpu_dataset = cuda_dataset.cpu()
    assert same_device(another_cpu_dataset.device, CPU)
