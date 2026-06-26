from matrix import Matrix

a = [[[1] for i in range(3)] for j in range(2)]
a = Matrix(a)

def print_matrix(egg): 
	for i in egg.elements:
		for j in i:
			print(j, end=' ')
		print()


print_matrix(a)
print()
a = a.transpose()
print()
print_matrix(a)

