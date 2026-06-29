from matrix import Matrix

x1 = [[1, 2, 3, 4, 5]]
w1 = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]
b1 = [[1, 2, 3, 4, 5]]
w2 = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]
b2 = [[0.5]]
true = [[1, 1, -1, -1, 1]]

x1 = Matrix(x1)
w1 = Matrix(w1)
b1 = Matrix(b1)
w2 = Matrix(w2)
b2 = Matrix(b2)
true = Matrix(true, has_grad = False)

l1 = x1 @ w1 + b1
l1_activated = l1.relu()
l2 = l1_activated @ w2 + b2
l2_activated = l2.relu()
loss = l2_activated.max_margin_loss(true)

print(loss.elements)
loss.backwards()
