from typing import Sequence
from unittest.mock import Mock

import pytest

from pearl.common import NodeValueType


@pytest.fixture
def make_parents_with_domain_sizes():
    def _make_parents_with_domain_sizes(domain_sizes: Sequence[int]):
        """
        :return a list of mock parents with value type CATEGORICAL and the desired domain sizes.
        """
        return [
            Mock(
                domain_size=ds, value_type=Mock(return_value=NodeValueType.CATEGORICAL)
            )
            for ds in domain_sizes
        ]

    return _make_parents_with_domain_sizes
