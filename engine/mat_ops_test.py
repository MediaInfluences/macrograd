from matrix import Matrix

a = [[1 for i in range(3)] for j in range(2)]
b = [[1 for i in range(2)] for j in range(3)]
d = [[1, 2, 3]]
e = [[1], [2], [3]]
f = [[1, 2, 3] for i in range(3)]

a = Matrix(a)
b = Matrix(b)
c = Matrix([[5]])
d = Matrix(d)
e = Matrix(e)
f = Matrix(f)

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

print("\n(1,1) + (n,m)")
print("\nc")
print_matrix(c)

print("\na")
print_matrix(a)

print("\nc + a")
test = c + a

print("\nresult")
print_matrix(test)

print("\n(1,m) + (n,m)")
print("\nd")
print_matrix(d)

print("\na")
print_matrix(a)

print("\nd + a")
test = d + a

print("\nresult")
print_matrix(test)

print("\n(n,m) + (1,m)")
print("\na")
print_matrix(a)

print("\nd")
print_matrix(d)

print("\na + d")
test = a + d 

print("\nresult")
print_matrix(test)

print("\n(n,m) + (n,1)")
print("\nf")
print_matrix(f)

print("\ne")
print_matrix(e)


print("\nf + e")
test = f + e 

print("\nresult")
print_matrix(test)

print("\n(n,1) + (n,m)")
print("\ne")
print_matrix(e)

print("\nf")
print_matrix(f)


print("\ne + f")
test = e + f 

print("\nresult")
print_matrix(test)


print("Hadamar Product Test")

print("\n(n,m) * (1,1)")
print("\na")
print_matrix(a)

print("\nc")
print_matrix(c)

print("\na * c")
test = a * c

print("\nresult")
print_matrix(test)

print("\n(1,1) * (n,m)")
print("\nc")
print_matrix(c)

print("\na")
print_matrix(a)

print("\nc * a")
test = c * a

print("\nresult")
print_matrix(test)

print("\n(1,m) * (n,m)")
print("\nd")
print_matrix(d)

print("\na")
print_matrix(a)

print("\nd * a")
test = d * a

print("\nresult")
print_matrix(test)

print("\n(n,m) * (1,m)")
print("\na")
print_matrix(a)

print("\nd")
print_matrix(d)

print("\na * d")
test = a * d 

print("\nresult")
print_matrix(test)

print("\n(n,m) * (n,1)")
print("\nf")
print_matrix(f)

print("\ne")
print_matrix(e)

print("\nf * e")
test = f * e 

print("\nresult")
print_matrix(test)

print("\n(n,1) * (n,m)")
print("\ne")
print_matrix(e)

print("\nf")
print_matrix(f)

print("\ne * f")
test = e * f 

print("\nresult")
print_matrix(test)


print("Matrix Multiplication Testing")
print("\n(1,m) @ (m,m)")
print("\nd")
print_matrix(d)

print("\nf")
print_matrix(f)

print("d @ f")
test = d @ f
print_matrix(test)

print("\n(m,m) @ (m, 1)")
print("\nf")
print_matrix(f)

print("\ne")
print_matrix(e)

print("f @ e")
test = f @ e 
print_matrix(test)

print("\n(1,m) @ (m, 1)")
print("\nd")
print_matrix(d)

print("\ne")
print_matrix(e)

print("d @ e")
test = d @ e 
print_matrix(test)
