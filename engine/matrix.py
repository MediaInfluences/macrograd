class Matrix():
	def __init__(self, elements: list[list[int | float]], _inputs = (),  _op = ''):
		assert isinstance(elements, list), "elements param must be of type list[list[int | float]]"
		for i in elements:
			for j in i:
				assert isinstance(j, list), "elements param must be of type list[list[int | float]]"
				for k in j:
					assert isinstance(k, (int | float)), "elements param must be of type list[list[int | float]]"
				assert len(j) == len(elements[0][0]), "elements param must be a non-jagged"

		self.elements = elements
		self.grad = 0
		self._dims = self._dimensions()
		self._inputs = _inputs
		self._op = _op	
		self._backward = lambda : None

		
	def _dimensions(self) -> tuple[int, int]:
		return (len(self.elements), len(self.elements[0]))
	
	#Optimize this further for funsies (Maybe for the C prog later on look into bitwise ops)
	def _should_broadcast(lhs_dims: tuple(int, int), rhs_dims: tuple(int, int)) -> str | bool:
		lhs_a_one = 1 if lhs_dims[0] == 1 else 0
		lhs_b_one = 1 if lhs_dims[1] == 1 else 0
		rhs_a_one = 1 if rhs_dims[0] == 1 else 0
		rhs_b_one = 1 if rhs_dims[1] == 1 else 0
		lhrh_a_match = 1 if lhs_dims[0] == rhs_dims[0] else 0
		lhrh_b_match = 1 if lhs_dims[1] == rhs_dims[1] else 0
		LHS = 0
		RHS = 0

		if lhs_a_one:
			if lhrh_b_match:
				return 'LHS' if not rhs_a_one
			return 'LHS' if lhs_b_one
		
		elif not lhs_a_one:
			if lhrh_a_match:
				return 'RHS' if (not lhrh_b_match) and (lhs_b_one or rhs_b_one)
			return 'RHS' if rhs_a_one and rhs_b_one

		return False
	
	
	#Probably something funky will arive here, note for when you debug backprop
	#Found it, holy handling cases that will never exist maxxing
	#Holy need to redo this thing to be right
	def _broadcast(self, dims: tuple[int, int]) -> Matrix:
		assert isinstance(dims, tuple), "dims param must be tuple[int, int]"

		out = self
		lhs_a_one = 1 if self._dims[0] == 1 else 0
		rhs_b_one = 1 if self._dims[1] == 1 else 0
		
		horizontal_broadcast = None
		vertical_broadcast = None

		for dimension in dims:
			assert isinstance(dimension, int), "dims param must be tuple[int, int]"

		if lhs_a_one:
			horizontal_broadcast = [[1] for i in range(dims[0])]
			horizontal_broadcast =  Matrix(horizontal_broadcast)
			out = horizontal_broadcast @ out		
			out = Matrix(out.elements, (horizontal_broadcast, self), '@')
	
		if lhs_b_one:
			vertical_broadcast = [1 for i in range(dims[1])]
			vertical_broadcast = Matrix(horizontal_broadcast)
			out = out @ vertical_broadcast
			out = Matrix(out.elements, (self, vertical_broadcast), '@')
		
		out = Matrix(out.elements, (self, ), 'broadcast')
		ones_matrix = None
	
		if lhs_a_one:
			if lhs_b_one:
				ones_matrix = horizontal_broadcast @ vertical_broadcast
				break

			ones_matrix = horizontal_broadcast 

		if lhs+b_one:
			ones_matrix = vertical_broadcast

		transposed_ones_matrix = ones_matrix.transpose()
		def _backward():
			self.grad += out.grad @ transposed_ones_matrix
		
		out._backward = _backward
		return out 


	def transpose(self) -> Matrix:
		transpose = [[self.elements[i][j] for i  in range(self._dims[0])] for j in range(self._dims[1])]
		out = Matrix(transpose, (self, ), 'T')
		
		def _backward():
			pass
		
		out._backward = _backward
		return out

		
	def hadamar_sum(self, other: Matrix) -> Matrix:
		assert(isinstance(other, Matrix)), "Both operands of a hadamar sum must be a matrix"
		maybe = should_broadcast(self._dims, other._dims)
		assert(maybe, True), f"Cannot perform a Hadamar Sum on elements of dim {self._dims} and {other._dims}"
		
		broadcasted = None
		
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			out = Matrix([xi + yi for xi, yi in zip(broadcast._elements, other._elements)], (broadcasted, other), 'hadamar sum')
		else:
			broadcasted = other._broadcast(other._dims)
			out = Matrix([xi + yi for xi, yi in zip(broadcast._elements, other._elements)], (self, broadcasted), 'hadamar sum')

		def _backward():
			pass

		out._backward = _backward
		return out
		

	def hadamar_product(self, other: Matrix) -> Matrix:
		assert(isinstance(other, Matrix)), "Both operands of a hadamar product must be type Matrix"
		maybe = should_broadcast(self._dims, other._dims)
		assert(maybe, True), f"Cannot perform a Hadamar Product on elements of dim {self._dims} and {other._dims}"
		
		broadcasted = None
		
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			out = Matrix([xi * yi for xi, yi in zip(broadcast._elements, other._elements)], (broadcasted, other), 'hadamar product')
		else:
			broadcasted = other._broadcast(other._dims)
			out = Matrix([xi * yi for xi, yi in zip(broadcast._elements, other._elements)], (self, broadcasted), 'hadamar product')

		def _backward():
			pass

		out._backward = _backward
		return out

	def __matmul__(self, other: Matrix) -> Matrix:
		#if this works, holy aura
		out = Matrix([[[sum(self.elements[i][j] * other.elements[j][k])] for j in range(self._dims[1]) for k in range(other._dims[0])] for i in range(self._dims[0])], (self, other), '@')

		def _backwards:
			pass

		out._backward = _backward
		return out
	

	def relu(self) -> Matrix:
		out  = Matrix([[i if i > 0 else 0 for i in self.elements[j]] for j in self.elements], (self, ), 'reLU')
		
		def _backward():
			pass
			
		out._backward = _backward
		return out			

	
	def max_margin_loss(self, truth: Matrix) -> Matrix:
		assert(isinstance(truth, Matrix), "True values must be of type Matrix"
		loss = Matrix(sum(max(0, 1 - y_true * y_pred) for y_pred, y_true in zip(pred, true) for pred, true in zip(self.elements, truth.elements)), (self, truth), 'max margin loss')
		
		def _backward():
			pass
		
		loss._backward = _backward
		return loss
	
	def backwards(self) -> None:
		topo = []
		
		def _construct(v):
			for child in v._inputs:
				_construct(v)
			topo.append(v)
		
		for node in reversed(topo):
			node.backward()
