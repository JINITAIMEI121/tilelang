"""T.Pipelined rejects invalid manual order/stage metadata."""

import pytest

import tilelang.ascend.language as T


def make_kernel(order, stage):
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1), T.SimtVF(threads=8):
            for i in T.Pipelined(8, order=order, stage=stage):
                A[i] = A[i] + 1

    return main


@pytest.mark.parametrize(
    "order, stage",
    [
        ([-1], [0]),
        ([0], [-1]),
        ([0], [0, 1]),
        ([0, 1], [0]),
    ],
)
def test_invalid_pipelined_metadata_rejected(order, stage):
    with pytest.raises(ValueError, match="T.Pipelined"):
        make_kernel(order, stage)
