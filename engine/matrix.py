"""Realized I designed _should_broadcast in the wrong way. FAAAAAH"""

class Matrix():
	def __init__(self, elements: list[list[int | float]], _inputs = (),  _op = ''):
		assert isinstance(elements, list), "elements param must be of type list[list[int | float]]"
		for i in elements:
			for j in i:
				assert isinstance(j, (int | float)), "elements param must be of type list[list[int | float]]"
				assert len(i) == len(elements[0]), "elements param must be a non-jagged"

		self.elements = elements
		self.grad = 0
		self._dims = self._dimensions()
		self._inputs = _inputs
		self._op = _op	
		self._backward = lambda: None

		
	def _dimensions(self) -> tuple[int, int]:
		return (len(self.elements), len(self.elements[0]))
	

	#Optimize this further for funsies (Maybe for the C prog later on look into bitwise ops
	def _should_broadcast(self, dims: tuple[int, int]) -> str | bool:
		lhs_a_one = 1 if self._dims[0] == 1 else 0
		lhs_b_one = 1 if self._dims[1] == 1 else 0
		rhs_a_one = 1 if dims[0] == 1 else 0
		rhs_b_one = 1 if dims[1] == 1 else 0
		lhrh_a_match = 1 if self._dims[0] == dims[0] else 0
		lhrh_b_match = 1 if self._dims[1] == dims[1] else 0
		LHS = 0
		RHS = 0

		if lhs_a_one:
			if lhrh_b_match:
				if not rhs_a_one:
					return 'LHS'
				if not rhs_a_one:
					return 'LHS' 
			if lhs_b_one:
				return 'LHS'		
		elif not lhs_a_one:
			if lhrh_a_match:
				if (not lhrh_b_match) and (lhs_b_one or rhs_b_one):
					return 'RHS'		
			if rhs_a_one and rhs_b_one:
				return 'RHS'
		return False
			
		
	def _broadcast(self, dims: tuple[int, int]):
		assert isinstance(dims, tuple), "dims param must be tuple[int, int]"
		for dimension in dims:
			assert isinstance(dimension, int), "dims param must be tuple[int, int]"
		assert ((self._dims[0] == 1) or (self._dims[1] == 1)), f"Cannot broadcast dims {self._dims} to {dims}" 

		if self._dims[0] == 1:
			horizontal_broadcast = [[1] for i in range(dims[0])]
			horizontal_broadcast =  Matrix(horizontal_broadcast)
			out = horizontal_broadcast @ self		
			
			def _backward():
				self.grad += horizontal_broadcast.transpose() @ out.grad
	
		elif self._dims[1] == 1:
			vertical_broadcast = [[1 for i in range(dims[1])]]
			vertical_broadcast = Matrix(vertical_broadcast)
			out = self @ vertical_broadcast
			
			def _backward():
				self.grad += out.grad @ vertical_broadcast.transpose() 
		
		else:
			return self
	
		out._backward = _backward
		return out 


	def transpose(self):
		transpose = [[self.elements[i][j] for i  in range(self._dims[0])] for j in range(self._dims[1])]
		out = Matrix(transpose, (self, ), 'T')
		
		def _backward():
			pass
		
		out._backward = _backward
		return out

		
	def hadamar_sum(self, other):
		assert(isinstance(other, Matrix)), "Both operands of a hadamar sum must be a matrix"
		maybe = self._should_broadcast(other._dims)
		assert isinstance(maybe, str), f"Cannot perform a Hadamar Sum on elements of dim {self._dims} and {other._dims}"
		
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			if (other._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(other._dims) 
				out = Matrix([xi + yi for xi, yi in zip(bbroadcasted.elements, other.elements)], (bbroadcasted, other), '+')
			else:
				out = Matrix([xi + yi for xi, yi in zip(broadcasted.elements, other.elements)], (broadcasted, other), '+')
		else:
			broadcasted = other._broadcast(self._dims)
			if (self._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(self._dims)
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(bbroadcasted.elements, self.elements)]
				out = Matrix(result, (self, broadcasted), '+')
			else:
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(broadcasted.elements, self.elements)]
				out = Matrix(result, (self, broadcasted), (self, broadcasted), '+')

		def _backward():
			pass

		out._backward = _backward
		return out
		

	def hadamar_product(self, other):
		assert(isinstance(other, Matrix)), "Both operands of a hadamar product must be type Matrix"
		maybe = self._should_broadcast(other._dims)
		assert isinstance(maybe, str), f"Cannot perform a Hadamar Product on elements of dim {self._dims} and {other._dims}"
			
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			if (other._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(other._dims) 
				out = Matrix([xi * yi for xi, yi in zip(bbroadcasted.elements, other.elements)], (bbroadcasted, other), '*')
			else:
				out = Matrix([xi * yi for xi, yi in zip(broadcasted.elements, other.elements)], (broadcasted, other), '*')
		else:
			broadcasted = other._broadcast(self._dims)
			if (self._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(self._dims)
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(bbroadcasted.elements, self.elements)]
				out = Matrix(result, (self, broadcasted), '*')
			else:
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(broadcasted.elements, self.elements)]
				out = Matrix(result, (self, broadcasted), (self, broadcasted), '*')

		def _backward():
			pass

		out._backward = _backward
		return out		
	
	def relu(self):
		out  = Matrix([[i if i > 0 else 0 for i in self.elements[j]] for j in self.elements], (self, ), 'reLU')
		
		def _backward():
			pass
			
		out._backward = _backward
		return out			

	
	def max_margin_loss(self, truth):
		assert(isinstance(truth, Matrix)), "True values must be of type Matrix"
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
			node._backward()

		
	def __matmul__(self, other):
		assert self._dims[1] == other._dims[0], f"Cannot perform matmul on operands of dims {self._dims} and {other._dims}" 
		elements = [[sum(self.elements[i][j] * other.elements[j][k] for j in range(self._dims[1])) for k in range(other._dims[1])] for i in range(self._dims[0])]	
		out = Matrix(elements, (self, other), '@')
	
		def _backward():
			pass

		out._backward = _backward
		return out

	def __add__(self, other):
		return self.hadamar_sum(other)

	def __radd__(self, other):
		return self.hadamar_sum(other)

	def __mul__(self, other):
		return self.hadamar_product(other)

	def __rmul__(self, other):
		return self.hadamar_product(other)
