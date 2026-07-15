from engine.matrix import Matrix
import pytest

ZERO_TEST = [
	("nm", (2,2), [[0, 0], [0, 0]]),
	("1m", (1,2), [[0, 0]]),
	("n1", (2,1), [[0], [0]]),
	("11", (1,1), [[0]])
]

@pytest.mark.parametrize(
	"dims, result",
	[(d,r) for _, d, r in ZERO_TEST],
	ids = [label for label, _, _ in ZERO_TEST],
)
def test_zeros(dims, result):
	assert Matrix.zeros(dims).elements == result


ONES_TEST = [
	("nm", (2,2), [[1, 1], [1, 1]]),
	("1m", (1,2), [[1, 1]]),
	("n1", (2,1), [[1], [1]]),
	("11", (1,1), [[1]])
]

@pytest.mark.parametrize(
	"dims, result",
	[(d,r) for _, d, r in ONES_TEST],
	ids = [label for label, _, _ in ONES_TEST],
)
def test_zeros(dims, result):
	assert Matrix.zeros(dims).elements == result


ZEROS_TEST = [
	("nm", (2,2), [[0, 0], [0, 0]]),
	("1m", (1,2), [[0, 0]]),
	("n1", (2,1), [[0], [0]]),
	("11", (1,1), [[0]])
]

@pytest.mark.parametrize(
	"dims, result",
	[(d,r) for _, d, r in ZEROS_TEST],
	ids = [label for label, _, _ in ZEROS_TEST],
)
def test_zeros(dims, result):
	assert Matrix.zeros(dims).elements == result


UNIFORM_TEST = [
	("nm", (2,2)),
	("1m", (1,2)),
	("n1", (2,1)),
	("11", (1,1))
]

@pytest.mark.parametrize(
	"dims",
	[d for _, d in UNIFORM_TEST],
	ids = [label for label, _ in UNIFORM_TEST],
)
def test_uniform(dims):
	assert Matrix.uniform(dims)._dims == dims
