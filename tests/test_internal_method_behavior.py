from engine.matrix import Matrix
import pytest

def test_internal_mathod_dimensions():
	test = Matrix([[7, 7, 7]])
	assert test._dimensions() == (1, 3)


SHOULD_BROADCAST_TEST = [
	("nm_nm", (2,3), (2,3), 'NONEED'),
	("1m_nm", (1,3), (2,3), 'LHS'),
	("nm_1m", (2,3), (1,3), 'RHS'),
	("n1_nm", (2,1), (2,3), 'LHS'),
	("nm_n1", (2,3), (2,1), 'RHS'),
	("11_nm", (1,1), (3,3), 'LHS'),
	("nm_11", (2,3), (1,1), 'RHS'),
	("nm_xy", (2,3), (3,3), False)
]

@pytest.mark.parametrize(
	"matrix_dims, bcast_dims, result",
	[(m,b,g) for _, m, b, g in SHOULD_BROADCAST_TEST],
	ids = [label for label, _, _, _ in SHOULD_BROADCAST_TEST],
)
def test_internal_method_should_broadcast(matrix_dims, bcast_dims, result):
	test = Matrix.ones(matrix_dims)
	should_bcast = test._should_broadcast(bcast_dims)
	assert should_bcast == result


BROADCAST_TEST = [
	("1m_nm", (1,3), (2,3)),
	("n1_nm", (2,1), (2,3)),
	("11_nm", (1,1), (2,3))
]

@pytest.mark.parametrize(
	"matrix_dims, bcast_dims",
	[(m,b) for _, m, b in BROADCAST_TEST],
	ids = [label for label, _, _ in BROADCAST_TEST],
)
def test_internal_method_broadcast(matrix_dims, bcast_dims):
	test = Matrix.ones(matrix_dims)
	bcasted_test = test._broadcast(bcast_dims)
	assert bcasted_test.elements == Matrix.ones(bcast_dims).elements


def test_internal_method_broadcast_fail():
	test = Matrix.ones((2,3))
	
	with pytest.raises(ValueError):
		test._broadcast((3,3))


BROADCAST_GRAD_TEST = [
	("1m_nm", (1,3), (2,3), [[2, 2, 2]]),
	("n1_nm", (2,1), (2,3), [[3], [3]]),
	("11_nm", (1,1), (2,3), [[6]])
]

@pytest.mark.parametrize(
	"matrix_dims, bcast_dims, grad",
	[(m,b,g) for _, m, b, g in BROADCAST_GRAD_TEST],
	ids = [label for label, _, _, _ in BROADCAST_GRAD_TEST],
)
def test_internal_method_broadcast_backward(matrix_dims, bcast_dims, grad):
	test = Matrix.ones(matrix_dims)
	bcasted_test = test._broadcast(bcast_dims)
	bcasted_test.backwards()
	assert test.grad.elements == grad
