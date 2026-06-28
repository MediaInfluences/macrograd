from matrix import Matrix

a = [[1 for i in range(3)] for j in range(2)]
b = [[1 for i in range(2)] for j in range(3)]
d = [[1, 2, 3]]
e = [[1], [2], [3]]
 
a = Matrix(a)
b = Matrix(b)
c = Matrix([[5]])
d = Matrix(d)
e = Matrix(e)

def print_matrix(egg): 
	for i in egg.elements:
		for j in i:
			print(j, end=' ')
		print("\n")

print("Transpose Test")

print_matrix(a)
a = a.transpose()

print_matrix(a)
a = a.transpose()

print_matrix(a)

print("a")
print_matrix(a)

print("b")
print_matrix(b)

print("a @ b")
test  = a @ b
print_matrix(test)

print("Hadamar Sum Test")

print("\n(n,m) + (1,1)")
print("\na")
print_matrix(a)

print("\nc")
print_matrix(c)

print("\na + c")
test = a + c

print("\nresult")
print_matrix(test)

print("\n(n,m) + (1,1)")
print("\na")
print_matrix(a)

print("\nc")
print_matrix(c)

print("\na + c")
test = a + c
print_matrix(test)


