"""T.copy rejects regions that start at a negative index."""

import pytest

import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            T.copy(A[-8:], B)

    return main


def test_copy_negative_source_offset_rejected():
    with pytest.raises(ValueError, match="negative"):
        make_kernel()
