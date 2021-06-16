import pytest
import torch


@pytest.mark.parametrize("event_dims", [(()), ((2,)), ((2, 3))])
def test_default_mapping_is_identity(tensor_index_mapping, size, event_dims):
    csi = tensor_index_mapping(size)
    cpd = torch.rand(size + event_dims)
    assert torch.equal(csi.map(cpd), cpd)


@pytest.mark.parametrize("event_dims", [(()), ((2,)), ((2, 3))])
@pytest.mark.parametrize(
    "index_range, canonical_index",
    [
        ((1, slice(None), slice(None)), (1, 0, 0)),
        ((1, 2, slice(None)), (1, 2, 0)),
        ((1, 2, 3), (1, 2, 3)),
    ],
)
def test_tensor_index_mapping(
    tensor_index_mapping, size, index_range, canonical_index, event_dims
):
    csi = tensor_index_mapping(size)
    csi.add_mapping(index_range, canonical_index)
    cpd = torch.rand(size + event_dims)
    cpd_with_csi = csi.map(cpd)
    assert torch.eq(cpd_with_csi[index_range], cpd[canonical_index]).all()
