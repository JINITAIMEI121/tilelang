"""T.Pipelined manual schedule metadata validation."""

import pytest

import tilelang.ascend.language as T


def test_pipelined_rejects_negative_order():
    with pytest.raises(ValueError, match="non-negative"):
        T.Pipelined(8, order=[-1], stage=[0])


def test_pipelined_rejects_negative_stage():
    with pytest.raises(ValueError, match="non-negative"):
        T.Pipelined(8, order=[0], stage=[-1])


def test_pipelined_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        T.Pipelined(8, order=[0], stage=[0, 1])
    with pytest.raises(ValueError, match="same length"):
        T.Pipelined(8, order=[0, 1], stage=[0])
