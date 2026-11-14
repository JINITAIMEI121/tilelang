"""T.Persistent rejects non-positive scheduling parameters."""

import pytest

import tilelang.ascend.language as T


def make_kernel(wave_size=1, group_size=8, num_stages=0):
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            for t in T.Persistent([8], wave_size, 0, group_size=group_size, num_stages=num_stages):
                A[t] = A[t] + 1

    return main


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"wave_size": -1}, "wave_size"),
        ({"group_size": -1}, "group_size"),
        ({"num_stages": -2}, "num_stages"),
    ],
)
def test_invalid_persistent_parameters_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_kernel(**kwargs)
