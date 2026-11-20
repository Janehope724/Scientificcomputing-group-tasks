import numpy as np
# coefficient matrix and right- hand side (boundaries 100 and 20)
A = np.array([[ 2,- 1, 0, 0, 0],
	[- 1, 2,- 1, 0, 0],
	[ 0,- 1, 2,- 1, 0],
	[ 0, 0,- 1, 2,- 1],
	[ 0, 0, 0,- 1, 2]], dtype=float)

b = np.array([100, 0, 0, 0, 20], dtype=float)
def jacobi(A,b,tol=1e-6, max_iter=1000):
	x = np.zeros_like(b)
	for k in range(1, max_iter + 1):
		x_new = np.zeros_like(x)
		for i in range(len(b)):
			s = sum(A[i, j] * x[j] for j in range(len(b)) if j != i)
			x_new[i] = (b[i] - s) / A[i, i]	# only old values used
		if np.max(np.abs(x_new - x)) < tol:
			return x_new, k
		x = x_new
	return x, max_iter
def gauss_seidel(A, b, tol=1e-6, max_iter=1000):
	x = np.zeros_like(b)
	for k in range(1, max_iter + 1):
		old = x.copy()
		for i in range(len(b)):
			s = sum(A[i, j] * x[j] for j in range(len(b)) if j != i)
			x[i] = (b[i] - s) / A[i, i]	# new values used at once
		if np.max(np.abs(x - old)) < tol:
			return x, k
	return x, max_iter
xj, kj = jacobi(A, b)
xg, kg = gauss_seidel(A, b)
print('Jacobi:', np.round(xj, 3), 'iterations:', kj)
print('Gauss- Seidel:', np.round(xg, 3), 'iterations:', kg)
print('Direct solve:', np.round(np.linalg.solve(A, b), 3))