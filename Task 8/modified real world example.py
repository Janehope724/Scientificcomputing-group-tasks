import numpy as np
# Iterative solver functions (from the original program)
def jacobi(A, b, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)
    for k in range(1, max_iter + 1):
        x_new = np.zeros_like(x)
        for i in range(len(b)):
            s = sum(A[i, j] * x[j] for j in range(len(b)) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]      # only old values used
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


def gauss_seidel(A, b, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(len(b)):
            s = sum(A[i, j] * x[j] for j in range(len(b)) if j != i)
            x[i] = (b[i] - s) / A[i, i]          # new values used immediately
        if np.max(np.abs(x - x_old)) < tol:
            return x, k
    return x, max_iter

# REAL-WORLD APPLICATION: 3-node electrical circuit
A2 = np.array([[10, -2,  1],
               [-2, 10, -3],
               [-1, -3, 10]], dtype=float)   # conductance matrix (S)
b2 = np.array([12, 10, 8], dtype=float)      # injected currents (A)

v, k = gauss_seidel(A2, b2)
print('Node voltages (V):', np.round(v, 4), 'iterations:', k)

# Compare with Jacobi
vj, kj = jacobi(A2, b2)
print('Jacobi node voltages (V):', np.round(vj, 4), 'iterations:', kj)


# Changed input: stronger source at node 3 (8 A -> 20 A)

b3 = np.array([12, 10, 20], dtype=float)
v3, k3 = gauss_seidel(A2, b3)
print('New node voltages (V):', np.round(v3, 4), 'iterations:', k3)