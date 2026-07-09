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

lr = Matrix([[0.5]], has_grad=False)

for i in range(100):
	l1 = x1 @ w1 + b1
	l1_activated = l1.relu()
	l2 = l1_activated @ w2 + b2
	loss = l2.cross_entropy_loss(true)
	print(loss.elements)
	loss.backwards(True)


	with Matrix.no_grad():
		w1 -= lr * w1.grad
		w1.grad = Matrix.zeros(w1._dims, has_grad=False)
		b1 -= lr * b1.grad
		b1.grad = Matrix.zeros(b1._dims, has_grad=False)
		w2  -= lr * w2.grad
		w2.grad = Matrix.zeros(w2._dims, has_grad=False)
		b2 -= lr * b2.grad
		b2.grad = Matrix.zeros(b2._dims, has_grad=False)

#Decide implicit versus exoplicit scalar conversions and their implications
#Make finite diff method to be able to confirm deny these results of grad calcs
#Implement detatch (new Matrix, no inputs or ops, and has_grad is False
#Figure out why the error w dividing by zero came up for lr = 2 when doing math (probably fp stuff as usual)
print(loss.elements)
