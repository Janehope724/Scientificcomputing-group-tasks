import numpy as np
# ORIGINAL: adjacency matrix of a 4-node network-
A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 1],
              [0, 1, 1, 0]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)
i = np.argmax(eigenvalues.real)
lam = eigenvalues[i].real
vec = np.abs(eigenvectors[:, i].real)
importance = vec / vec.sum()

print('Dominant eigenvalue:', round(lam, 4))
for node, score in zip('ABCD', importance):
    print(node, round(score, 4))

v = eigenvectors[:, i].real
print('Verification error:', np.linalg.norm(A @ v - lam * v))
# REAL-WORLD APPLICATION: PageRank on 4 web pages
# Column j shows how page j shares its vote among pages it links to
L = np.array([[0,    0,   1,   0.5],
              [1/3,  0,   0,   0  ],
              [1/3,  0.5, 0,   0.5],
              [1/3,  0.5, 0,   0  ]])

#  No damping 
w, V = np.linalg.eig(L)
i = np.argmax(w.real)
rank = np.abs(V[:, i].real)
print('PageRank (no damping):', np.round(rank / rank.sum(), 4))

#  With damping factor d = 0.85 
d = 0.85
M = d * L + (1 - d) / 4 * np.ones((4, 4))
w, V = np.linalg.eig(M)
i = np.argmax(w.real)
rank = np.abs(V[:, i].real)
print('PageRank (damping 0.85):', np.round(rank / rank.sum(), 4))