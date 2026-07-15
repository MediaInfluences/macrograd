"""
ToDo:
	- Define repr
	- Write test suite
	- Write sum function (maybe?)
	- Fix the arithmatic / logic mistakes in some of the dunders
	- Verify the grads were undoubled and there is no issues with finite diffs
	- Test the context manager so people can use the with torch.no_grad pattern (or @torch.no_grad)
	- Test uniform
	- Test detatch
	- Regretting design choices
	- Read through my code and see what needs to be shifted for the Pytorch mentality

Fin:
	- Change assert stuff to raise stuff
	- update the backwards for the matrix ops (or in general) to check for the has_grad thing in the _backward step to prevent None += from happening
	- Check the dunders and 2 param funcs to make it so when both of the inputs have has_grad = False the parent will also inertly has a has_grad = False
	- Make it so the rest of the one param funcs keep their parents has_grad
	- One hot should not have a grad, it cannot be differentiated and for the most part why would you want it to be

Deffered:
	- Think about optimizing the broadcast check
"""
from __future__ import annotations
import random 
import math

class Matrix():
	yesgrad = True	
	def __init__(self, elements: list[list[int | float]] | int | float, _inputs = (),  _op = '', has_grad = True):
		if isinstance(elements, (int | float)):
			self.elements = [[elements]]
			_inputs = ()
			_op = ''
			has_grad = False

		else:
			if not isinstance(elements, list):
				raise TypeError(f"elements param must be of type list[list[int | float]];  is type {type(elements)}")
			for i in elements:
				if not isinstance(elements, list):
					raise TypeError(f"elements param must be of type list[list[int | float]];  is type {type(elements)}")
			
				for j in i:
					if not isinstance(j, (int | float)):
						raise TypeError(f"elements param must be of type list[list[int | float]], is type {type(j)}")
					if len(i) != len(elements[0]):
						raise ValueError("elements param must be non-jagged")
			self.elements = elements
		
		if not Matrix.yesgrad:
			_inputs = () 
			_op = ''
			has_grad = False
	
		self._dims = self._dimensions()
		self._inputs = _inputs
		self._op = _op	
		self.has_grad = has_grad
		self._backward = lambda: None

		if self.has_grad:
			self.grad = Matrix.zeros(self._dims, has_grad = False)
		else:
			self.grad = None


#---------- Internal Methods ----------#
	def _dimensions(self) -> tuple[int, int]:
		return (len(self.elements), len(self.elements[0]))

	#emberasiiiing	
	def _should_broadcast(self, dims: tuple[int, int]) -> str | bool:
		if self._dims == dims:
			return 'NONEED'

		if self._dims[0] == dims[0]:
			if self._dims[1] == 1:
				return 'LHS'
			elif dims[1] == 1:
				return 'RHS'
			else:	
				return False

		if self._dims[1] == dims[1]:
			if self._dims[0] == 1:
				return 'LHS'
			elif dims[0] == 1:
				return 'RHS'
			else:	
				return False

		if self._dims == (1,1):
			return 'LHS'
		
		if dims == (1,1):
			return 'RHS'
	
		return False

	def _broadcast(self, dims: tuple[int, int]):
		if not isinstance(dims, tuple):
			raise TypeError(f"dims param must be tuple[int, int]; is type: {type(dims)}")

		for dimension in dims:
			if not isinstance(dimension, int):
				raise TypeError(f"dims param must be tuple[int, int]; is type {type(dims)}")

		if not (((self._dims[0] == 1) or (self._dims[1] == 1))):
			raise ValueError(f"Cannot broadcast dims {self._dims} to {dims}")

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
		
		out = Matrix(out.elements, (self, ), 'broadcast', has_grad = self.has_grad)

		def _backward():
			ograd = out.grad
			if dim0_is1:
				ograd = horizontal_broadcast.transpose() @ ograd	

			if dim1_is1:
				ograd = ograd @ vertical_broadcast.transpose()

			if self.has_grad:
				self.grad += ograd
		
		out._backward = _backward
		return out 


#---------- Slice-Of-Life Methods ----------#
	@staticmethod	
	def zeros(dims: tuple[int, int], has_grad = True):
		elements = [[0 for j in range(dims[1])] for i in range(dims[0])]
		out = Matrix(elements, has_grad = has_grad)
		return out

	
	@staticmethod
	def ones(dims: tuple[int, int], has_grad = True):
		elements = [[1 for j in range(dims[1])] for i in range(dims[0])]
		out = Matrix(elements, has_grad = has_grad)
		return out


	@staticmethod
	def uniform(dims: tuple[int, int], ab = (-1, 1), has_grad = True):
		elements = [[random.uniform(ab[0], ab[1]) for j in range(dims[1])] for i in range(dims[0])]
		out = Matrix(elements, has_grad = has_grad)
		return out


	#one-hot should not have grad -> is transformation to indexing
	@staticmethod
	def one_hot(matrix: Matrix, num_classes: int):
		if num_classes <= 0:
			raise ValueError(f"num_classes must be greater than 0")

		if matrix._dims[0] != 1:
			raise ValueError(f"Macrograd only supports row vectors as input for the one_hot operation; has dims: {matrix._dims}")

		for i in matrix.elements[0]:
			if not isinstance(i, int):
				raise TypeError("Cannot call one_hot on matrices with non-integer elements")

		one_hot = Matrix.zeros((matrix._dims[1], num_classes), has_grad = False)
		
		n = 0
		for i in matrix.elements[0]:
			one_hot.elements[n][i] = 1
			n += 1
		
		return one_hot
	
	
#---------- Matrix Operations ----------#
	#Things that act on only one input should respect their parent's has_grad choice, no reason a no_grad gets transposed and suddenly wants a grad	
	def transpose(self):
		transpose = [[self.elements[i][j] for i in range(self._dims[0])] for j in range(self._dims[1])]
		out = Matrix(transpose, (self, ), 'T', has_grad = self.has_grad)
		
		def _backward():
			if self.has_grad:
				self.grad += out.grad.transpose()
		
		out._backward = _backward
		return out

		
	def hadamar_sum(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		maybe = self._should_broadcast(other._dims)
		if not isinstance(maybe, str):
			raise ValueError(f"Cannot perform a Hadamar Sum on elements of dim {self._dims} and {other._dims}")

		either_has_grad = self.has_grad or other.has_grad
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			if (other._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(other._dims) 
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(bbroadcasted.elements, other.elements)]

				def _backward():
					if bbroadcasted.has_grad:	
						bbroadcasted.grad += out.grad

					if other.has_grad:
						other.grad += out.grad

				out = Matrix(result, (bbroadcasted, other), '+', has_grad = either_has_grad)
			
			else:
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(broadcasted.elements, other.elements)]
			
				def _backward():
					if broadcasted.has_grad:
						broadcasted.grad += out.grad

					if other.has_grad:
						other.grad += out.grad

				out = Matrix(result, (broadcasted, other), '+', has_grad = either_has_grad)

		elif maybe == 'RHS':
			broadcasted = other._broadcast(self._dims)
			if (self._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(self._dims)
				result = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, bbroadcasted.elements)]
			
				def _backward():
					if self.has_grad:
						self.grad += out.grad

					if bbroadcasted.has_grad:
						bbroadcasted.grad += out.grad

				out = Matrix(result, (self, bbroadcasted), '+', has_grad = either_has_grad)
			
			else:
				result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, broadcasted.elements)]
				
				def _backward():
					if self.has_grad:
						self.grad += out.grad

					if broadcasted.has_grad:
						broadcasted.grad += out.grad

				out = Matrix(result, (self, broadcasted), '+', has_grad = either_has_grad)

		else:		
			result  = [[xi + yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, other.elements)]

			out = Matrix(result, (self, other), '+', has_grad = either_has_grad)

			def _backward():
				if self.has_grad:
					self.grad += out.grad

				if other.has_grad:
					other.grad += out.grad

		out._backward = _backward
		return out
		

	def hadamar_product(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		maybe = self._should_broadcast(other._dims)
		if not isinstance(maybe, str):
			raise ValueError(f"Cannot perform a Hadamar Product on elements of dim {self._dims} and {other._dims}")

		either_has_grad = self.has_grad or other.has_grad
		if maybe == 'LHS':
			broadcasted = self._broadcast(other._dims)
			if (other._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(other._dims) 
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(bbroadcasted.elements, other.elements)]

				def _backward():
					if bbroadcasted.has_grad:
						bbroadcasted.grad += other * out.grad

					if other.has_grad:
						other.grad += bbroadcasted * out.grad

				out = Matrix(result, (bbroadcasted, other), '*', has_grad = either_has_grad)
			
			else:
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(broadcasted.elements, other.elements)]
			
				def _backward():
					if broadcasted.has_grad:
						broadcasted.grad += other * out.grad

					if other.has_grad:
						other.grad += broadcasted * out.grad

				out = Matrix(result, (broadcasted, other), '*', has_grad = either_has_grad)

		elif maybe == 'RHS':
			broadcasted = other._broadcast(self._dims)
			if (self._dims != broadcasted._dims):
				bbroadcasted = broadcasted._broadcast(self._dims)
				result = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, bbroadcasted.elements)]
			
				def _backward():
					if self.has_grad:
						self.grad += bbroadcasted * out.grad

					if bbroadcasted.has_grad:
						bbroadcasted.grad += self * out.grad

				out = Matrix(result, (self, bbroadcasted), '*', has_grad = either_has_grad)
			
			else:
				result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, broadcasted.elements)]
				
				def _backward():
					if self.has_grad:
						self.grad += broadcasted * out.grad

					if broadcasted.has_grad:
						broadcasted.grad += self * out.grad

				out = Matrix(result, (self, broadcasted), '*', has_grad = either_has_grad)

		else:		
			result  = [[xi * yi for xi, yi in zip(i, j)] for i, j in zip(self.elements, other.elements)]

			out = Matrix(result, (self, other), '*', has_grad = either_has_grad)

			def _backward():
				if self.has_grad:
					self.grad += other * out.grad

				if other.has_grad:
					other.grad += self * out.grad

		out._backward = _backward
		return out

	
#------- Activation Functions ----------#
	
	def relu(self):
		result = [[self.elements[i][j] if self.elements[i][j] > 0 else 0 for j in range(self._dims[1])] for i in range(self._dims[0])]
		out = Matrix(result, (self, ), 'reLU', has_grad = self.has_grad)
		
		def _backward():
			if self.has_grad:
				result = [[1 if self.elements[i][j] > 0 else 0 for j in range(self._dims[1])] for i in range(self._dims[0])]
				relu_gate = Matrix(result, has_grad = False)
				self.grad += relu_gate * out.grad			
			
		out._backward = _backward
		return out


#---------- Loss Functions ----------#

	def cross_entropy_loss(self, truth):
		if not isinstance(truth, Matrix):
			raise TypeError(f"True values must be of type Matrix; is type: {type(truth)}")

		expd = self.exp()
		sum_expd = [[sum(expd.elements[xi][xj] for xj in range(expd._dims[1]))] for xi in range(expd._dims[0])]	
		sum_expd = Matrix(sum_expd)
		probs = expd / sum_expd.transpose()
		
		out = probs * truth
		out = [sum(out.elements[xi][xj] for xj in range(out._dims[1])) for xi in range(out._dims[0])]

		nll = 0.0
		for survivor in out:
			nll += -math.log(survivor)	
		
		out = Matrix([[nll/len(truth.elements[0])]], (self, ), 'cross entropy loss', has_grad = self.has_grad)	

		def _backward():
			if self.has_grad:
				self.grad += (probs - truth)/len(truth.elements[0]) * out.grad
		
		out._backward = _backward
		return out 


	def max_margin_loss(self, truth):
		if not isinstance(truth, Matrix):
			raise TypeError(f"True values must be of type Matrix; is type: {type(truth)}")
		out = Matrix([[sum(max(0, 1 - y_true * y_pred) for pred, true in zip(self.elements, truth.elements) for y_pred, y_true in zip(pred, true))]], (self, truth), 'max margin loss', has_grad = self.has_grad)
		
		def _backward():
			if self.has_grad:
				self.grad += Matrix([[0 if 1 - y_true * y_pred < 0 else -1 * y_true for y_true, y_pred in zip(i, j)] for i, j in zip(truth.elements, self.elements)])			
				
		out._backward = _backward
		return out


#---------- Backpropagation & Friends  ----------#

	@staticmethod
	def no_grad():
		return NoGrad()


	def detatch(self):
		return Matrix(self.elements, (), '', has_grad = False)


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
		self.grad = Matrix.ones(self._dims, has_grad = False)
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
		out = Matrix(result, (self, ), 'exp', has_grad = self.has_grad)
	
		def _backward():
			if self.has_grad:
				self.grad += self * out.grad

		out._backward = _backward
		return out

	
	def log(self):
		result = [[math.log(self.elements[xi][xj]) for xj in range(self._dims[1])] for xi in range(self._dims[0])]
		out = Matrix(result, (self, ), 'log', has_grad = self.has_grad)
		
		def _backward():
			if self.has_grad:	
				self.grad += (self**-1) * out.grad
		
		out._backward = _backward	
		return out	


	def __matmul__(self, other):
		if self._dims[1] != other._dims[0]:
			raise ValueError(f"Cannot perform matmul on operands of dims {self._dims} and {other._dims}")
 
		elements = [[sum(self.elements[i][j] * other.elements[j][k] for j in range(self._dims[1])) for k in range(other._dims[1])] for i in range(self._dims[0])]	
		either_has_grad = self.has_grad or other.has_grad
		out = Matrix(elements, (self, other), '@', has_grad = either_has_grad)
	
		def _backward():
			if self.has_grad:
				self.grad += out.grad @ other.transpose()
			if other.has_grad:	
				other.grad += self.transpose() @ out.grad

		out._backward = _backward
		return out


	def __add__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_sum(other)


	def __radd__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_sum(other)

	
	def __iadd__(self, other):
		if self.has_grad and Matrix.yesgrad:
			raise RuntimeError("in-place updates on a grad-tracked matrix while autograd is recording")
		self.elements = [[self.elements[i][j] + other.elements[i][j] for j in range(self._dims[1])] for i in range(self._dims[0])]
		return self


	def __neg__(self):
		return self * -1	


	def __sub__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self + (other * -1)

	
	def __isub__(self, other):
		if self.has_grad and Matrix.yesgrad:
			raise RuntimeError("in-place updates on a grad-tracked matrix while autograd is recording")
		self.elements = [[self.elements[i][j] - other.elements[i][j] for j in range(self._dims[1])] for i in range(self._dims[0])]
		return self

	
	def __rsub__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return other + (self * -1)


	def __mul__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_product(other)


	def __rmul__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self.hadamar_product(other)

	
	def __pow__(self, n):
		if not isinstance(n, (int, float)):
			 raise TypeError(f"power only supports int or float exponent values; is type {type(n)}")

		result = [[self.elements[xi][xj] ** n for xj in range(self._dims[1])] for xi in range(self._dims[0])]
		out = Matrix(result, (self, ), 'pow', has_grad = self.has_grad)
	
		def _backward():
			if self.has_grad:
				self.grad += n * self**(n-1) * out.grad
	
		out._backward = _backward
		return out

	
	def __truediv__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return self * other**-1


	def __rtruediv__(self, other):
		other = other if isinstance(other, Matrix) else Matrix(other)
		return other * self**-1


#---------- NoGrad ----------#
	
class NoGrad:
	def __enter__(self):
		self.prev = Matrix.yesgrad
		Matrix.yesgrad = False
	
	def __exit__(self, exc_type, exc_value, traceback):
		Matrix.yesgrad = self.prev
