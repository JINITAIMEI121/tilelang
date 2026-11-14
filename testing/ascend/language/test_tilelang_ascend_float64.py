"""float64 tensor buffers are rejected before device compilation on Ascend."""

import pytest

import tilelang.ascend.language as T
from tilelang.engine.lower import lower


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((8,), "float64"), B: T.Tensor((8,), "float64")):
        with T.Kernel(1), T.SimtVF(threads=8):
            for i in T.Parallel(8):
                A[i] = A[i] + B[i]

    return main


def test_float64_tensor_rejected():
    with pytest.raises(ValueError, match="float64"):
        lower(make_kernel(), target="ascend")
