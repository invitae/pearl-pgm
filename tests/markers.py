import os

import pytest
import torch

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cuda device is needed to run these tests",
)

slow = pytest.mark.skipif(
    not os.getenv("RUN_SLOW_TESTS", False),
    reason="enable slow running tests by setting 'RUN_SLOW_TESTS' environment variable",
)
