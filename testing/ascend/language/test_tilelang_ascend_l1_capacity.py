"""alloc_l1 rejects allocations that exceed the dav-3510 L1 capacity."""

import pytest

import tilelang.ascend.language as T


def test_alloc_l1_rejects_oversize_allocation():
    def make_kernel():
        @T.prim_func
        def main(A: T.Tensor((8, 8), "bfloat16")):
            with T.Kernel(1):
                huge = T.alloc_l1((1024, 1024, 1024), "bfloat16")  # 2 GiB
                T.copy(A, huge[0, 0:8, 0:8])

        return main

    with pytest.raises(ValueError, match="L1 capacity"):
        make_kernel()


def test_alloc_l1_accepts_fitting_allocation():
    @T.prim_func
    def main(A: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            tile = T.alloc_l1((64, 64), "bfloat16")
            T.copy(A, tile)

    assert main is not None
