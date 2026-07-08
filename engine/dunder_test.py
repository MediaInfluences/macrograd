from matrix import Matrix

def print_mat(matrix):
	for i in matrix.elements:
		for j in i:
			print(j, end=' ')
		print('\n')

a = [[5 for j in range(3)] for i in range(3)]
b = [[2 for j in range(1)] for i in range(3)]
c = [[2 for j in range(3)]]

a = Matrix(a)
b = Matrix(b)
c = Matrix(c)

test = a**2
print_mat(test)
test = b**2
print_mat(test)
test = c**2
print_mat(test)

test = a**2 / a

print_mat(test)

test = a.exp()
print_mat(test)

test = b.exp()
print_mat(test)

test = c.exp()
print_mat(test)

test = a - a

print_mat(test)

test = a - c 

print_mat(test)

test = -a 

print_mat(test)

test = a / a

print_mat(test)

test = a.exp()

print_mat(test)

test = Matrix.zeros((3,3))

print_mat(test)

test = Matrix.zeros((3,1))

print_mat(test)

test = Matrix.zeros((1,3))

print_mat(test)

print('#----Zeros within one_hot ----#')
test = Matrix.ones((1,3))
print_mat(test)

test = test.one_hot(3)
print('#----One Hot----#')
print_mat(test)

test = a.cross_entropy_loss(c)
print(test)

