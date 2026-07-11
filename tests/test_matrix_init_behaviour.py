from engine.matrix import Matrix

def test_matrix_init_when_element_scalar():
	test = Matrix(777)
	assert test.elements == [[777]]
	assert test._dims == (1, 1)
	assert test._inputs == ()
	assert test._op == ''
	assert test.has_grad is False
	assert test.grad is None


def test_matrix_init_when_element_noncalar():
	test = Matrix([[8, 6, 7, 5, 3, 0 , 9]])
	assert test.elements == [[8, 6, 7, 5, 3, 0, 9]]
	assert test._dims == (1, 7)
	assert test._inputs == ()
	assert test._op == ''
	assert test.has_grad is True
	assert test.grad == [[0, 0, 0, 0, 0, 0, 0]]
	assert isinstance(test.grad.grad, None)


def test_matrix_init_when_element_nested_nonscalars():
	test = Matrix([[8, 6, 7, 5, 3, 0 , 9]])
	assert test.elements == [[8, 6, 7], [5, 3, 0], [9, 9, 9]]
	assert test._dims == (3, 3)
	assert test._inputs == ()
	assert test._op == ''
	assert test.has_grad is True
	assert test.grad == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
	assert isinstance(test.grad.grad, None)

def test_matrix_init_when_element_is_not_list():
	with pytest.raises(TypeError):
		test = Matrix("I am (not) a list") 


def test_matrix_init_when_element_list_of_scalars():
	with pytest.raises(TypeError):
		test = Matrix([8, 6, 7, 5, 3, 0, 9])


def test_matrix_init_when_element_jagged():
	with pytest.raises(ValueError):
		test = Matrix([[8, 6, 7], [5, 3, 0], [9]])


def test_matrix_init_when_matrix_yesgrad_false():
	with Matrix.no_grad():
		test = Matrix([[7, 7, 7]])
		assert test.elements == [[7, 7, 7]]
		assert test._dims == (1, 3)
		assert test._inputs == ()
		assert test._op == ''
		assert test.has_grad is False
		assert test.grad is None
