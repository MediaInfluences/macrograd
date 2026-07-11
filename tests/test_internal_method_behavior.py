from engine.matrix import Matrix

def test_internal_mathod_dimensions_call():
	test = Matrix([[7, 7, 7]])
	assert test._dimensions == (1, 3)


def test_internal_method_should_broadcast_nm_nm():	
	test = Matrix([[7, 7, 7], [7, 7, 7]])
	assert test._should_broadcast((2,3)) == 'NONEED'
	

def test_internal_method_should_broadcast_1m_nm():	
	test = Matrix([[7, 7, 7]])
	assert test._should_broadcast((3,3)) == 'LHS'


def test_internal_method_should_broadcast_nm_1m():	
	test = Matrix([[7, 7, 7], [7, 7, 7]])
	assert test._should_broadcast((1,3)) == 'RHS'


def test_internal_method_should_broadcast_n1_nm():	
	test = Matrix([[7], [7], [7]])
	assert test._should_broadcast((3,3)) == 'LHS'


def test_internal_method_should_broadcast_nm_n1():
	test = Matrix([[7, 7, 7], [7, 7, 7]])
	assert test._should_broadcast((2,1)) == 'RHS'


def test_internal_method_should_broadcast_11_nm():	
	test = Matrix(1)
	assert test._should_broadcast((3,3)) == 'LHS'


def test_internal_method_should_broadcast_nm_11():	
	test = Matrix([[7, 7, 7], [7, 7, 7], [7, 7, 7]])
	assert test._should_broadcast((1,1)) == 'RHS'

	
def test_internal_method_should_broadcast_nm_xy():	
	test = Matrix([[7, 7, 7], [7, 7, 7], [7, 7, 7]])
	assert test._should_broadcast((2,2)) is False


def test_internal_method_broadcast_1m_nm():
	test = Matrix([[7, 7, 7]])
	assert test._broadcast((2,3)).elements == [[7, 7, 7], [7, 7, 7]]


def test_internal_method_broadcast_n1_nm():
	test = Matrix([[7], [7], [7]])
	assert test._broadcast((3,2)).elements == [[7, 7], [7, 7], [7, 7]]


def test_internal_method_broadcast_11_nm():
	test = Matrix(7)
	assert test._broadcast((3,3)).elements == [[7, 7, 7], [7, 7, 7], [7, 7, 7]]


def test_internal_method_broadcast_nm():
	test = Matrix([[7, 7, 7], [7, 7, 7]])
	
	with pytest.raises(ValueError):
		test._broadcast((3,3))


#Need to write gradient tests in scenarios above for broadcast	
