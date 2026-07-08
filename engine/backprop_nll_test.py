from matrix import Matrix

x1 = [[1, 2, 3, 4, 0]]
w1 = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]
b1 = [[1, 2, 3, 4, 5]]
w2 = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]
b2 = [[0.5]]
true = [[1, 3, 2, 4, 0]]

x1 = Matrix(x1)
x1 = x1.one_hot(5)
w1 = Matrix(w1)
b1 = Matrix(b1)
w2 = Matrix(w2)
b2 = Matrix(b2)
true = Matrix(true, has_grad = False)
true = true.one_hot(5)

lr = 0.5

for i in range(10):
	l1 = x1 @ w1 + b1
	l1_activated = l1.relu()
	l2 = l1_activated @ w2 + b2
	l2_activated = l2.relu()
	loss = l2_activated.cross_entropy_loss(true)
	print(loss.elements)
	loss.backwards(True)

#make lambda to pass in the update function or maybe learning rate by itself)
#make a uniform func to init weights

	w1 -= Matrix([[lr]], has_grad = False) * w1.grad
	w1.grad = Matrix.zeros(w1._dims)
	b1 -= Matrix([[lr]], has_grad = False) * b1.grad
	b1.grad = Matrix.zeros(b1._dims)
	w2  -= Matrix([[lr]], has_grad = False) * w2.grad
	w2.grad = Matrix.zeros(w2._dims)
	b2 -= Matrix([[lr]], has_grad = False) * b2.grad
	b2.grad = Matrix.zeros(b2._dims)
	

print(loss.elements)
