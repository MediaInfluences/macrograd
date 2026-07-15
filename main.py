from engine.matrix import Matrix

x1 = Matrix([[1, -1, 2, -2, 3, -3, 4, -4]])

w1 = Matrix.uniform((8,16))
w2 = Matrix.uniform((16,6))
w3 = Matrix.uniform((6,16))
w4 = Matrix.uniform((16,16))
w5 = Matrix.uniform((16,8))

b1 = Matrix.uniform((1,16))
b2 = Matrix.uniform((1,6))
b3 = Matrix.uniform((1,16))
b4 = Matrix.uniform((1,16))
b5 = Matrix.uniform((1,8))

truth = Matrix([[1, -1, 1, -1, 1, -1, 1, -1]], has_grad = False)
lr = 0.0075

for epoch in range(25):
	l1 = x1 @ w1 + b1
	act1 = l1.relu()
	l2 = act1 @ w2 + b2
	act2 = l2.relu()
	l3 = act2 @ w3 + b3
	act3 = l3.relu()
	l4 = act3 @ w4 + b4
	l5 = l4 @ w5 + b5
	loss = l5.max_margin_loss(truth)
	print(f"epoch: {epoch}     loss: {loss.elements}")
	loss.backwards(True)
	
	with Matrix.no_grad():
		w1 -= lr * w1.grad
		w2 -= lr * w2.grad
		w3 -= lr * w3.grad
		w4 -= lr * w4.grad
		w5 -= lr * w5.grad

		b1 -= lr * b1.grad
		b2 -= lr * b2.grad
		b3 -= lr * b3.grad
		b4 -= lr * b4.grad
		b5 -= lr * b5.grad

		w1.grad = Matrix.zeros((8,16))
		w2.grad = Matrix.zeros((16,6))
		w3.grad = Matrix.zeros((6,16))
		w4.grad = Matrix.zeros((16,16))
		w5.grad = Matrix.zeros((16,8))

		b1.grad = Matrix.zeros((1,16))
		b2.grad = Matrix.zeros((1,6))
		b3.grad = Matrix.zeros((1,16))
		b4.grad = Matrix.zeros((1,16))
		b5.grad = Matrix.zeros((1,8))
