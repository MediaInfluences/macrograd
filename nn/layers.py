from engine.matrix import Matrix
import random

class DenseLayer():
	def __init__(self, w_dims_0: int, w_dims_1: int):
		self.w_dims = (w_dims_0, w_dims_1)
		self.b_dims = (1, w_dims_1)
		self.w = Matrix([[random.uniform(-1, 1) for j in range(w_dims_1)] for i in range(w_dims_0)])
		self.b = Matrix([[random.uniform(-1, 1) for j in range(w_dims_1)]])

	
	def forward(self, inputs: Matrix):
		out = inputs @ self.w + self.b
		return out

	
	def update(self, lr: float):
		for i in range(self.w_dims[0]):
			for j in range(self.w_dims[1]):
				self.w.elements[i][j] += -1 * lr * self.w.grad.elements[i][j] 		

		self.w.grad = Matrix([[0 for j in range(self.w_dims[1])] for i in range(self.w_dims[0])])

		for i in range(self.b_dims[0]):
			for j in range(self.b_dims[1]):
				self.b.elements[i][j] += -1 * lr * self.b.grad.elements[i][j] 		

		self.b.grad = Matrix([[0 for j in range(self.b_dims[1])] for i in range(self.b_dims[0])])	

class ActivationFunc():
	def relu(self, prev_layer):
		return prev_layer.relu()


class LossFunc():
	def __init__(self, truth: list[list[int|float]]):
		self.truth = Matrix(truth)

	
	def max_margin_loss(self, prev_layer):
		return prev_layer.max_margin_loss(self.truth)


class MLP():
	def __init__(self, input: list[list[int|float]], lr: float, _show_graph = False):	
		self.input = Matrix(input)	
		self.layers = []
		self.lr = lr
		self._show_graph = _show_graph	

	
	def add_layer(self, layer ):
		self.layers.append(layer)

	
	def forward(self):
		xn = self.layers[0].forward(self.input)
		for i in range(len(self.layers) - 2):
			if isinstance(self.layers[i+1], DenseLayer):
				xn = self.layers[i+1].forward(xn)
			
			if isinstance(self.layers[i+1], ActivationFunc):
				xn = self.layers[i+1].relu(xn)
	
		return self.layers[-1].max_margin_loss(xn)

	
	def backwards(self):
		for layer in self.layers:
			if isinstance(layer, DenseLayer):
				layer.update(self.lr)


	def epoch(self, num: int):
		for i in range(num):
			loss = self.forward()		
			print(f"\nepoch {i} loss: {loss.elements}")
			loss.backwards(self._show_graph)
			self.backwards()
