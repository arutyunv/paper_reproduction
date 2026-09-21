import numpy as np

from tasks.multiplication import MultiplicationTask


def test_multiplication():
    task = MultiplicationTask(
        n_digits=5,
        multiplier_digits=3,
        seed=42,
    )

    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8])

    result = task._multiply_digits(a, b)

    # 12345 * 678 = 8369910
    # fixed width = 5 + 3 = 8
    assert result.tolist() == [
        0, 8, 3, 6, 9, 9, 1, 0
    ]


def test_shapes():
    task = MultiplicationTask(
        n_digits=5,
        multiplier_digits=3,
        seed=42,
    )

    item = task.sample()[0]

    # 5 digits + MUL + 3 digits + EQ
    assert len(item.prompt) == 10

    # 5 + 3
    assert len(item.answer) == 8


def test_long_multiplication():
    task = MultiplicationTask(
        n_digits=35,
        multiplier_digits=3,
        seed=42,
    )

    item = task.sample()[0]

    assert len(item.prompt) == 40
    assert len(item.answer) == 38
    assert item.metadata["n_digits"] == 35
    assert item.metadata["length_group"] == "LONG"


def test_collate():
    task = MultiplicationTask(
        n_digits=5,
        multiplier_digits=3,
        seed=42,
    )

    items = task.sample(32)

    x, y = task.collate(
        items,
        block_size=task.min_block_size,
    )

    assert x.shape == (32, task.min_block_size)
    assert y.shape == (32, task.min_block_size)

    # Each example has exactly 8 supervised answer positions.
    assert np.all((y != -1).sum(axis=1) == 8)

