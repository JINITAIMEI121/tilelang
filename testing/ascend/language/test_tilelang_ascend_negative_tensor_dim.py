"""Negative tensor dimensions are rejected at annotation time."""

import pytest

import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((-4,), "float32")):
        with T.Kernel(1):
            A[0] = A[0] + 1

    return main


def test_negative_tensor_dimension_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        make_kernel()
