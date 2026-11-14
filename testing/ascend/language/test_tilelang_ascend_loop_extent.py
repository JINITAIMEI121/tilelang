"""Ascend loop builders reject negative extents at the API boundary."""

import pytest

import tilelang.ascend.language as T


def make_parallel_kernel():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1), T.SimtVF(threads=8):
            for i in T.Parallel(-8):
                A[i] = A[i] + 1

    return main


def make_serial_kernel():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            for i in T.serial(-8):
                A[i] = A[i] + 1

    return main


def make_unroll_kernel():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            for i in T.unroll(-8):
                A[i] = A[i] + 1

    return main


@pytest.mark.parametrize("factory, match", [
    (make_parallel_kernel, "T.Parallel"),
    (make_serial_kernel, "T.serial"),
    (make_unroll_kernel, "T.unroll"),
])
def test_negative_loop_extent_rejected(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
