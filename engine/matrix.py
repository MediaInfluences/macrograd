"""
ToDo:
	- Add uniform (easily init weights)
	- Add a context manager so people can use the with torch.no_grad pattern (or @torch.no_grad)
	- Fix the double tracking issue for backprop stuff (either with no_grad or some other method)
	- Read through my code and see what needs to be shifted for the Pytorch mentality

Fin:
	- Regretting design choices

Working theory is to change how the ops interact with scalar values instead of just making them into Matrix objs
"""
import math

#Make has_grad be like an init flag, 
class Matrix():
	yesgrad = True
	def __init__(self, elements: list[list[int | float]] | int | float, _inputs = (),  _op = '', has_grad = True):
		if isinstance(elements, (int | float)):
			self.elements = [[elements]]
		else:
			assert isinstance(elements, list), f"elements param must be of type list[list[int | float]], is type {type(elements)}"
			for i in elements:
				for j in i:
					assert isinstance(j, (int | float)), f"elements param must be of type list[list[int | float]], is type {type(j)}"
					assert len(i) == len(elements[0]), "elements param must be a non-jagged"
			self.elements = elements

		self._dims = self._dimensions()
		self._inputs = _inputs
		self._op = _op	
		self.has_grad = has_grad

		if not Matrix.yesgrad:
			self._inputs = () 
			self._op = ''
			self.has_grad = False
			
		self._backward = lambda: None

		if self.has_grad:
			init_grad = [[0 for j in range(self._dims[1])] for i in range(self._dims[0])]
			self.grad = Matrix(init_grad, has_grad = False)
		else:
			self.grad = None


#---------- Internal Methods ----------#
		
	def _dimensions(self) -> tuple[int, int]:
		return (len(self.elements), len(self.elements[0]))
	

	def _should_broadcast(self, dims: tuple[int, int]) -> str | bool:
		lhs_a_one = 1 if self._dims[0] == 1 else 0
		lhs_b_one = 1 if self._dims[1] == 1 else 0
		rhs_a_one = 1 if dims[0] == 1 else 0
		rhs_b_one = 1 if dims[1] == 1 else 0
		lhrh_a_match = 1 if self._dims[0] == dims[0] else 0
		lhrh_b_match = 1 if self._dims[1] == dims[1] else 0
		LHS = 0
		RHS = 0

		if lhs_a_one: 					#(1,?) (?,?)
			if lhrh_b_match:			#(1,m) (?,m)
				if lhs_b_one:			#(1,1) (?,1)
					if not rhs_a_one:	#(1,1) (n,1) ; n!=1
						return 'LHS'	#True
					return 'NONEED'		#(1,1) (1,1) -> False
				if not rhs_a_one:		#(1,m) (n,m) ; n!=1
					return 'LHS'		#True
				return 'NONEED'			#(1,m) (1,m) -> No Need
			if lhs_b_one:				#(1,1) (?,m) ; m!=1
				return 'LHS'			#True
			if rhs_a_one:				#(1,y) (1,x) ; y!=1
				if rhs_b_one:			#(1,y) (1,1) ; y!=1
					return 'RHS'		#True	
				return False			#(1,y) (1,x) ; x,y != 1
			return False				#(1,x) (?,y) -> False

		if not lhs_a_one:				#(n,?) (?,?) ; n!=1
			if lhrh_a_match:			#(n,?) (n,?)
				if not lhrh_b_match:		#(n,x) (n,y)
					if lhs_b_one:		#(n,1) (n,y)
						return 'LHS'	#True
					if rhs_b_one:		#(n,x) (n,1) 		
						return 'RHS'	#True		
					return False		#(n,x) (n,y) -> False
				return 'NONEED' 		#(n,m) (n,m) ; n!=1 -> False
			if rhs_a_one:				#(x,?) (1,?) ; x!= 1
				if lhrh_b_match:		#(x,m) (1,m) ; x!= 1
					return 'RHS'		#True
				if rhs_b_one:			#(n,x) (1,1) ; x!=1
					return 'RHS'		#True
				return False			#(x,n) (1,m) ; m!=1  -> False
			return False				#(x,?) (y,?) ; x,y != 1 -> False
	
		
	def _broadcast(self, dims: tuple[int, int]):
		assert isinstance(dims, tuple), "dims param must be tuple[int, int]"
		for dimension in dims:
			assert isinstance(dimension, int), "dims param must be tuple[int, int]"
		assert ((self._dims[0] == 1) or (self._dims[1] == 1)), f"Cannot broadcast dims {self._dims} to {dims}"
		dim0_is1 = 1 if self._dims[0] == 1 else 0
		dim1_is1 = 1 if self._dims[1] == 1 else 0
		
		out = self
		if dim0_is1:
			horizontal_broadcast = [[1] for i in range(dims[0])]
			horizontal_broadcast =  Matrix(horizontal_broadcast)
			out = horizontal_broadcast @ out		
			
		if dim1_is1:
			vertical_broadcast = [[1 for i in range(dims[1])]]
			vertical_broadcast = Matrix(vertical_broadcast)
			out = out @ vertical_broadcast
		
		out = Matrix(out.elements, (self, ), 'broadcast')

		def _backward():
			ograd = out.grad
			if dim0_is1:
				ograd = horizontal_broadcast.transpose() @ ograd	
			if dim1_is1:
				ograd = ograd @ vertical_broadcast.transpose()
			self.grad += ograd
		
		out._backward = _backward
		return out 


#---------- Slice-Of-Life Methods ----------#
	@staticmethod
	def no_grad():
		return NoGrad()

	
	@staticmethod	
	def zeros(dims: tuple[int, int], has_grad = True):
		elements = [[0 for j in range(dims[1])] for i in range(dims[0])]
		out = Matrix(elements) if has_grad else Matrix(elements, has_grad=False)
		return out

	
	@staticmethod
	def ones(dims: tuple[int, int], has_grad = True):
		elements = [[1 for j in range(dims[1])] for i in range(dims[0])]
		out = Matrix(elements) if has_grad else Matrix(elements, has_grad=False)
		return out

	
	def one_hot(self, num_classes: int, has_grad = True):
		assert self._dims[0] == 1, "Macrograd only supports row vectors as input for the one_hot operation"
		for i in self.elements[0]:
			assert isinstance(i, int), "Cannot call one_hot on matrices with non-integer elements"

		one_hot = Matrix.zeros((self._dims[1], num_classes), has_grad=has_grad)
		
		n = 0
		for i in self.elements[0]:
			one_hot.elements[n][i] = 1
			n += 1
		
		return one_hot

 
#---------- Matrix Operations ----------#
	
	def transpose(self):
		transpose = [[self.elements[i][j] for i in range(self._dims[0])] for j in range(self._dims[1])]
		out = Matrix(transpose, (self, ), 'T')
		
		def _backward():
			self.grad += out.grad.transpose()
		
		out._backward = _backward
		return out

		
	def hadamar_sum(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		maybe = self._should_broadcast(other._dims)
		assert isinstance(maybe, str), f"Cannot perform a Hadamar Sum on elements of dim {self._dims} and {other._dims}"
		
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			if (other._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(other._dims) 
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(bbroadcasted.elements, other.elements)]

				def _backward():
					bbroadcasted.grad += out.grad
					other.grad += out.grad

				out = Matrix(result, (bbroadcasted, other), '+')
			
			else:
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(broadcasted.elements, other.elements)]
			
				def _backward():
					broadcasted.grad += out.grad
					other.grad += out.grad
					
				out = Matrix(result, (broadcasted, other), '+')

		elif maybe == 'RHS':
			broadcasted = other._broadcast(self._dims)
			if (self._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(self._dims)
				result = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, bbroadcasted.elements)]
			
				def _backward():
					self.grad += out.grad
					bbroadcasted.grad += out.grad

				out = Matrix(result, (self, bbroadcasted), '+')
			
			else:
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, broadcasted.elements)]
				
				def _backward():
					self.grad += out.grad
					broadcasted.grad += out.grad

				out = Matrix(result, (self, broadcasted), '+')

		else:		
			result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, other.elements)]
			out = Matrix(result, (self, other), '+')

			def _backward():
				self.grad += out.grad
				other.grad += out.grad

		out._backward = _backward
		return out
		

	def hadamar_product(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		maybe = self._should_broadcast(other._dims)
		assert isinstance(maybe, str), f"Cannot perform a Hadamar Product on elements of dim {self._dims} and {other._dims}"
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			if (other._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(other._dims) 
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(bbroadcasted.elements, other.elements)]
				out = Matrix(result, (bbroadcasted, other), '*')
				
				def _backward():
					bbroadcasted.grad += other * out.grad
					other.grad += bbroadcasted * out.grad
			
			else:
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(broadcasted.elements, other.elements)]
				out = Matrix(result, (broadcasted, other), '*')
				
				def _backward():
					broadcasted.grad += other * out.grad
					other.grad += broadcasted * out.grad

		elif maybe == 'RHS':
			broadcasted = other._broadcast(self._dims)
			if (self._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(self._dims)
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, bbroadcasted.elements)]
				out = Matrix(result, (self, bbroadcasted), '*')
				
				def _backward():
					self.grad += bbroadcasted * out.grad
					bbroadcasted.grad += self * out.grad


			else:
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, broadcasted.elements)]
				out = Matrix(result, (self, broadcasted), '*')
		
				def _backward():
					self.grad += broadcasted * out.grad
					broadcasted.grad += self * out.grad
		
		else:
			result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, other.elements)]
			out = Matrix(result, (self, other), '*')

			def _backward():
				self.grad += other * out.grad
				other.grad += self * out.grad

		out._backward = _backward
		return out


#------- Activation Functions ----------#
	
	def relu(self):
		result = [[self.elements[i][j] if self.elements[i][j] > 0 else 0 for j in range(self._dims[1])] for i in range(self._dims[0])]
		out = Matrix(result, (self, ), 'reLU')
		
		def _backward():
			result = [[1 if self.elements[i][j] > 0 else 0 for j in range(self._dims[1])] for i in range(self._dims[0])]
			relu_gate = Matrix(result, has_grad = False)
			self.grad += relu_gate * out.grad			
		
		out._backward = _backward
		return out


#---------- Loss Functions ----------#
	
	def cross_entropy_loss(self, truth):
		assert(isinstance(truth, Matrix)), "True values must be of type Matrix"
		expd = self.exp()
		sum_expd = [[sum(expd.elements[xi][xj] for xj in range(expd._dims[1]))] for xi in range(expd._dims[0])]	
		sum_expd = Matrix(sum_expd)
		probs = expd / sum_expd.transpose()
		
		out = probs * truth
		out = [sum(out.elements[xi][xj] for xj in range(out._dims[1])) for xi in range(out._dims[0])]

		nll = 0.0
		for survivor in out:
			nll += -math.log(survivor)	

		out = Matrix(nll/len(truth.elements[0]), (self, ), 'cross entropy loss')	

		def _backward():
			self.grad += (probs - truth)/len(truth.elements[0]) * out.grad
		
		out._backward = _backward
		return out 


	def max_margin_loss(self, truth):
		assert(isinstance(truth, Matrix)), "True values must be of type Matrix"
		loss = Matrix([[sum(max(0, 1 - y_true * y_pred) for pred, true in zip(self.elements, truth.elements) for y_pred, y_true in zip(pred, true))]], (self, truth), 'max margin loss')
		
		def _backward():
			self.grad += Matrix([[0 if 1 - y_true * y_pred < 0 else -1 * y_true for y_true, y_pred in zip(i, j)] for i, j in zip(truth.elements, self.elements)])			
				
		loss._backward = _backward
		return loss


#---------- Backpropagation & Friends  ----------#

	def backwards(self, show_graph = False) -> None:
		topo = []
		visited = set()
		
		def _construct(v):
			if v not in visited:
				visited.add(v)
				for child in v._inputs:
					_construct(child)
				topo.append(v)

		_construct(self)
		self.grad = Matrix([[1.0]], has_grad = False)
		for node in reversed(topo):
			if node.has_grad:
				node._backward()
		
		if show_graph:
			from graphviz import Digraph	

			def _tbl(M, title):
				cols = len(M[0])
				cells = "".join("<tr>" + "".join(f"<td>{x:.3g}</td>" for x in r) + "</tr>" for r in M)
				return (f"<table border='0' cellborder='1' cellspacing='0'>"
						f"<tr><td colspan='{cols}'><b>{title}</b></td></tr>{cells}</table>")

			def node_label(node):
				fwd = _tbl(node.elements, "forward")
				grd = (_tbl(node.grad.elements, "grad")
						if node.grad is not None else "<table><tr><td>—</td></tr></table>")
				return (f"<<table border='0' cellborder='0' cellspacing='10'>"
						f"<tr><td colspan='2'><b>{node._op or 'leaf'}  {node._dims}</b></td></tr>"
						f"<tr><td>{fwd}</td><td>{grd}</td></tr></table>>")

			dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
			for node in topo:
				dot.node(str(id(node)), label=node_label(node), shape="plaintext")
				if node._op:							
					op_id = str(id(node)) + node._op
					dot.node(op_id, label=node._op)
					dot.edge(op_id, str(id(node)))
				for child in node._inputs:	
					dot.edge(str(id(child)), str(id(node)) + node._op)

			dot.render("graph", view=True, cleanup=True)
	

#---------- Primitives ---------#

	def exp(self):
		result = [[math.exp(self.elements[xi][xj]) for xj in range(self._dims[1])] for xi in range(self._dims[0])]
		out = Matrix(result, (self, ), 'exp')
	
		def _backward():
			self.grad += self * out.grad

		out._backward = _backward
		return out

	
	def log(self):
		result = [[math.log(self.elements[xi][xj]) for xj in range(self._dims[1])] for xi in range(self._dims[0])]
		out = Matrix(result, (self, ), 'log')
		
		def _backward():
			self.grad += (self**-1) * out.grad
		
		out._backward = _backward	
		return out	


	def __matmul__(self, other):
		assert self._dims[1] == other._dims[0], f"Cannot perform matmul on operands of dims {self._dims} and {other._dims}" 
		elements = [[sum(self.elements[i][j] * other.elements[j][k] for j in range(self._dims[1])) for k in range(other._dims[1])] for i in range(self._dims[0])]	
		out = Matrix(elements, (self, other), '@')
	
		def _backward():
			self.grad += out.grad @ other.transpose()
			other.grad += self.transpose() @ out.grad

		out._backward = _backward
		return out


	def __add__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_sum(other)


	def __radd__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_sum(other)


	def __neg__(self):
		return self * Matrix(-1)	


	def __sub__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self + (other * Matrix(-1))

	
	def __rsub__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return other + (self * Matrix(-1))

	#Probably change how muls work with just scalar vals
	#Should show on operator graph, but should not have graph
	def __mul__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_product(other)


	def __rmul__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_product(other)

	
	def __pow__(self, n):
		assert isinstance(n, (int, float)), "power only supports int or float exponent values"
		curr_elements = Matrix(self.elements, has_grad = False)

		result = [[self.elements[xi][xj] ** n for xj in range(self._dims[1])] for xi in range(self._dims[0])]
		out = Matrix(result, (self, ), 'pow')
	
		def _backward():
			self.grad += (n * self / curr_elements) * out.grad
	
		out._backward = _backward
		return out

	
	def __truediv__(self, other):
		return self * other**-1


	def __rtruediv__(self, other):
		return self * other**-1


#---------- NoGrad ----------#
	
class NoGrad:
	def __enter__(self):
		self.prev = Matrix.yesgrad
		Matrix.yesgrad = False
	
	def __exit__(self, exc_type, exc_value, traceback):
		Matrix.yesgrad = self.prev
