import numpy as np
# adjacency matrix, node order is A, B, C, D
A = np.array([[0, 1, 1, 0],
	[1, 0, 1, 1],
	[1, 1, 0, 1],
	[0, 1, 1, 0]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
i = np.argmax(eigenvalues.real)		# index of largest eigenvalue
lam = eigenvalues[i].real
vec = np.abs(eigenvectors[:, i].real)
importance = vec / vec.sum()	# scores add up to 1
print('Dominant eigenvalue:', round(lam, 4))
for node, score in zip('ABCD', importance):
	print(node, round(score, 4))
# check that A v = lambda v really holds
v = eigenvectors[:, i].real
print('Verification error:', np.linalg.norm(A @ v - lam * v))